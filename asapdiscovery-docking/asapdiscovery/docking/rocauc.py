import math

import numpy as np
import pandas as pd


class Rock:
    """
    A class to calculate ROC-AUC results for different scoring functions for a
    dataset.
    """

    def __init__(
        self,
        df,
        score_name,
        rmsd_name="RMSD",
    ):
        self.df = df
        self.score_name = score_name
        self.rmsd_name = rmsd_name

        (
            self.total_poses,
            self.total_good_poses,
            self.total_bad_poses,
        ) = self.calc_data(self.df)

        self.auc = str
        self.auc_list = []

    def calc_data(self, df, rmsd_cutoff=2):
        """
        Given a dataset, return the six things we want to know

        Parameters
        ----------
        df: pd.DataFrame

        Returns
        -------

        """
        n_poses = len(df)
        n_good_poses = sum(df[self.rmsd_name] <= rmsd_cutoff)
        n_bad_poses = n_poses - n_good_poses

        return (
            n_poses,
            n_good_poses,
            n_bad_poses,
        )

    def calc_auc(self, false_positive_rates, true_positive_rates):
        """
        Calculates the area under the curve using the numpy function!
        Parameters
        ----------
        false_positive_rates
        true_positive_rates

        Returns
        -------

        """
        return np.trapz(x=false_positive_rates, y=true_positive_rates)

    def get_score_range(self, n_bins):
        """
        For some score in the data with arbitrary minima and maxima, return n
        evenly spaced points along the score.
        Returns
        -------

        """
        # Give a score with min and max values, create n_bins
        # cutoffs with the first cutoff being slightly below the
        # first value so that the first 0 point is included.
        min = self.df[self.score_name].min()
        max = self.df[self.score_name].max()
        delta = (max - min) / n_bins
        start = min - delta
        score_range = np.linspace(
            start,
            max,
            n_bins,
            endpoint=True,
        )
        return score_range

    def weird_division(self, numerator, denominator):
        """
        This function will divide the numerator by the denominator but will
        return zero if the denominator is zero.
        Parameters
        ----------
        numerator
        denominator

        Returns
        -------

        """
        return numerator / denominator if denominator else 0

    def get_auc_from_df(self, df=None, bootstrap=False, n_bins=10):
        self.score_range = self.get_score_range(n_bins)

        if df is None:
            # In the case that we're using the data saved to this class, we can
            # pull out the original totals
            df = self.df
            total_poses = self.total_poses
            total_good_poses = self.total_good_poses
            total_bad_poses = self.total_bad_poses

        else:
            # if a new dataframe is passed, that means we are bootstrapping,
            # in which case we need to re-calculate the 'self' totals
            assert type(df) == pd.DataFrame
            (
                total_poses,
                total_good_poses,
                total_bad_poses,
            ) = self.calc_data(df)

        true_positive_rates_poses = []  # same thing as recall
        false_positive_rates_poses = []
        self.precision_poses = []

        # I *think* this is faster than iterating through and making lists for
        # each thing but I don't actually know
        data = [
            self.calc_data(df[df[self.score_name] <= cutoff])
            for cutoff in self.score_range
        ]
        (
            n_poses_list,
            n_good_poses_list,
            n_bad_poses_list,
        ) = zip(*data)

        for idx in range(len(n_poses_list)):
            n_poses = n_poses_list[idx]
            n_good_poses = n_good_poses_list[idx]
            n_bad_poses = n_bad_poses_list[idx]

            true_positive_rates_poses.append(n_good_poses / total_good_poses)
            false_positive_rates_poses.append(n_bad_poses / total_bad_poses)

            if not bootstrap:
                # Don't care about bootstrapping these

                self.precision_poses.append(self.weird_division(n_good_poses, n_poses))

        if bootstrap:
            self.auc_list.append(
                self.calc_auc(false_positive_rates_poses, true_positive_rates_poses)
            )
        else:
            self.true_positive_rates_poses = true_positive_rates_poses
            self.false_positive_rates_poses = false_positive_rates_poses
            self.auc = self.calc_auc(
                false_positive_rates_poses, true_positive_rates_poses
            )
        return self.auc

    def get_bootstrapped_error_bars(self, n_bootstraps):

        # First, make sure we have calculated values for data
        self.get_auc_from_df(self.df, bootstrap=False)

        # Then bootstrap CVs
        self.auc_poses = [
            self.get_auc_from_df(self.df.sample(frac=1, replace=True), bootstrap=True)
            for n in range(n_bootstraps)
        ]

        # Make this list a numpy arrray so we can use some numpy functions
        auc_poses_array = np.array(self.auc_poses)

        # sort the array from smallest to largest AUC so we can use 95% confidence
        # interval numerically
        auc_poses_array.sort()

        # the 95% confidence interval includes everything but the bottom and
        # top 2.5% (0.025).
        # i.e. with 1000 sorted values, our lower CI bound is the 25th value and our
        # upper CI bound is the 975th value
        auc_poses_bounds = math.floor(len(auc_poses_array) * 0.025)

        # The CI's are reported as the difference (i.e. +/-) as opposed to the actual
        # values
        self.poses_ci = (
            auc_poses_array.mean() - auc_poses_array[auc_poses_bounds],
            auc_poses_array[-auc_poses_bounds] - auc_poses_array.mean(),
        )

    def get_tidy_df_for_figure(self):
        self.auc_poses_df = pd.DataFrame(
            {
                "True_Positive": self.true_positive_rates_poses,
                "False_Positive": self.false_positive_rates_poses,
                "Value": self.score_range,
                "Score_Type": self.score_name,
                "Precision": self.precision_poses,
            }
        )


class Rocks:
    """
    Class for analyzing docking data, for comparing from among different scoring
    functions.
    """

    def __init__(
        self,
        csv,
        score_list,
        rmsd_name,
        n_bins,
        n_bootstraps=None,
    ):
        # First save all the passed in variables
        self.csv = csv
        self.score_list = score_list
        self.n_bins = n_bins
        self.n_bootstraps = n_bootstraps
        self.rmsd_name = rmsd_name

        # Clean the dataframe
        self.clean_dataframe()

        # Make a dictionary of Rock objects
        self.build_rocks()

        # Calculate AUC
        self.get_aucs()

    def clean_dataframe(self):
        df = pd.read_csv(self.csv)
        df["POSIT_R"] = 1 - df["POSIT"]

        # TODO: Expose these hard-coded options
        self.df = df[(df["Chemgauss4"] < 100) & (df["RMSD"] < 20) & (df["RMSD"] > 0)]

    def build_rocks(self):
        self.rock_dict = {
            score_name: Rock(self.df, score_name)  # , self.rmsd_name, self.n_bins
            for score_name in self.score_list
            if score_name in self.df.columns
        }

    def get_aucs(self):
        """
        Calculates AUC for each ROC curve.

        Returns
        -------

        """
        for score_name, rock in self.rock_dict.items():
            rock.get_auc_from_df(n_bins=self.n_bins)
            rock.get_tidy_df_for_figure()
            self.rock_dict[score_name] = rock

    def combine_dfs(self):
        _ = [rock.get_tidy_df_for_figure() for rock in self.rock_dict.values()]
        poses_dfs = [rock.auc_poses_df for rock in self.rock_dict.values()]
        self.poses_df = pd.concat(poses_dfs)

    def get_auc_cis(self):

        lower_bound_list = []
        upper_bound_list = []
        auc_list = []
        for score_name, rock in self.rock_dict.items():
            print(score_name)
            rock.get_bootstrapped_error_bars(self.n_bootstraps)
            lower_bound_list.append(rock.poses_ci[0])
            upper_bound_list.append(rock.poses_ci[1])
            auc_list.append(rock.auc_poses[0])
        self.model_df = pd.DataFrame(
            {
                "Score_Type": self.score_list,
                "Lower_Bound": lower_bound_list,
                "AUC": auc_list,
                "Upper_Bound": upper_bound_list,
            }
        )
