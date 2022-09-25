import numpy as np
import math
import pandas as pd
import plotly.express as ex


class Rock:
    """
    Class for getting ROC-AUC results for different scoring functions for a dataset.
    """

    def __init__(
        self,
        df,
        score_name,
        rmsd_name,
        n_samples,
    ):
        self.df = df
        self.score_name = score_name
        self.rmsd_name = rmsd_name
        self.n_samples = n_samples
        self.get_score_range()

        (
            self.total_poses,
            self.total_good_poses,
            self.total_bad_poses,
            self.total_cmpds,
            self.total_good_cmpds,
            self.total_bad_cmpds,
        ) = self.calc_data(self.df)

        self.auc = str
        self.auc_list = []

    def calc_data(self, df):
        """
        Given a dataset, return the six things we want to know
        Parameters
        ----------
        df

        Returns
        -------

        """
        n_poses = len(df)
        n_good_poses = sum(df[self.rmsd_name] <= 2)
        n_bad_poses = n_poses - n_good_poses

        n_cmpds = len(set(df.Compound_ID))
        set_of_good_cmpds = set(df[df[self.rmsd_name] <= 2].Compound_ID)
        n_good_cmpds = len(set_of_good_cmpds)
        n_bad_cmpds = n_cmpds - n_good_cmpds

        return (
            n_poses,
            n_good_poses,
            n_bad_poses,
            n_cmpds,
            n_good_cmpds,
            n_bad_cmpds,
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

    def get_score_range(self):
        """
        For some score in the data with arbitrary minima and maxima, return n evenly spaced points along the score.
        Returns
        -------

        """
        self.score_range = np.linspace(
            self.df[self.score_name].min() - 1,
            self.df[self.score_name].max(),
            self.n_samples,
            endpoint=True,
        )

    def weird_division(self, n, d):
        return n / d if d else 0

    def get_auc_from_df(self, df=None, bootstrap=False):
        if df is None:
            ## In the case that we're using the data saved to this class, we can pull out the original totals
            df = self.df
            total_poses = self.total_poses
            total_good_poses = self.total_good_poses
            total_bad_poses = self.total_bad_poses
            total_good_cmpds = self.total_good_cmpds
            total_bad_cmpds = self.total_bad_cmpds

        else:
            ## if a new dataframe is passed, that means we are bootstrapping,
            ## in which case we need to re-calculate the 'self' totals

            (
                total_poses,
                total_good_poses,
                total_bad_poses,
                total_cmpds,
                total_good_cmpds,
                total_bad_cmpds,
            ) = self.calc_data(df)

        true_positive_rates_poses = []  ## same thing as recall
        false_positive_rates_poses = []
        self.precision_poses = []

        self.true_positive_rates_cmpds = []  ## same thing as recall
        # self.false_positive_rates_cmpds = []
        self.precision_cmpds = []

        ## I *think* this is faster than iterating through and making lists for each thing but I don't actually know
        data = [
            self.calc_data(df[df[self.score_name] <= cutoff])
            for cutoff in self.score_range
        ]
        (
            n_poses_list,
            n_good_poses_list,
            n_bad_poses_list,
            n_cmpds_list,
            n_good_cmpds_list,
            n_bad_cmpds_list,
        ) = zip(*data)

        for idx in range(len(n_poses_list)):
            n_poses = n_poses_list[idx]
            n_good_poses = n_good_poses_list[idx]
            n_bad_poses = n_bad_poses_list[idx]

            true_positive_rates_poses.append(n_good_poses / total_good_poses)
            false_positive_rates_poses.append(n_bad_poses / total_bad_poses)

            if not bootstrap:
                ## Don't care about bootstrapping these

                n_cmpds = n_cmpds_list[idx]
                n_good_cmpds = n_good_cmpds_list[idx]
                self.precision_poses.append(
                    self.weird_division(n_good_poses, n_poses)
                )
                self.true_positive_rates_cmpds.append(
                    n_good_cmpds / total_good_cmpds
                )
                ## this doesn't really make sense mathematically
                # self.false_positive_rates_cmpds.append(n_bad_cmpds / self.total_bad_cmpds)
                self.precision_cmpds.append(
                    self.weird_division(n_good_cmpds, n_cmpds)
                )

        if bootstrap:
            self.auc_list.append(
                self.calc_auc(
                    false_positive_rates_poses, true_positive_rates_poses
                )
            )
        else:
            self.true_positive_rates_poses = true_positive_rates_poses
            self.false_positive_rates_poses = false_positive_rates_poses
            self.auc = self.calc_auc(
                false_positive_rates_poses, true_positive_rates_poses
            )
        return self.auc

    def get_bootstrapped_error_bars(self, n_bootstraps):

        ## First, make sure we have calculated values for data
        self.get_auc_from_df(self.df, bootstrap=False)

        ## Then bootstrap CVs
        self.auc_poses = [
            self.get_auc_from_df(
                self.df.sample(frac=1, replace=True), bootstrap=True
            )
            for n in range(n_bootstraps)
        ]

        auc_poses_array = np.array(self.auc_poses)
        try:
            auc_poses_array.sort()
        except:
            print(auc_poses_array)

        auc_poses_bounds = math.floor(len(auc_poses_array) * 0.025)
        self.poses_ci = (
            auc_poses_array.mean() - auc_poses_array[auc_poses_bounds],
            auc_poses_array[-auc_poses_bounds] - auc_poses_array.mean(),
        )

    def get_df(self):
        self.auc_poses_df = pd.DataFrame(
            {
                "True_Positive": self.true_positive_rates_poses,
                "False_Positive": self.false_positive_rates_poses,
                "Value": self.score_range,
                "Score_Type": self.score_name,
                "Precision": self.precision_poses,
            }
        )
        self.auc_cmpds_df = pd.DataFrame(
            {
                "True_Positive": self.true_positive_rates_cmpds,
                "Value": self.score_range,
                "Score_Type": self.score_name,
                "Precision": self.precision_cmpds,
            }
        )


class Rocks:
    """
    Class for analyzing docking data, for comparing from among different scoring functions.
    """

    def __init__(
        self,
        csv,
        score_list,
        rmsd_name,
        n_samples,
        n_bootstraps=None,
    ):
        self.csv = csv
        self.score_list = score_list
        self.n_samples = n_samples
        self.n_bootstraps = n_bootstraps
        self.rock_dict = {}
        self.rmsd_name = rmsd_name

        self.clean_dataframe()
        self.build_rocks()
        # self.df = df

    def clean_dataframe(self):
        df = pd.read_csv(self.csv)
        df["POSIT_R"] = -df["POSIT"] + 1
        self.df = df[
            (df["Chemgauss4"] < 100) & (df["RMSD"] < 20) & (df["RMSD"] > 0)
        ]

    def build_rocks(self):
        for score_name in self.score_list:
            assert score_name in self.df.columns
            self.rock_dict[score_name] = Rock(
                self.df, score_name, self.rmsd_name, self.n_samples
            )

    def get_aucs(self):
        for score_name, rock in self.rock_dict.items():
            rock.get_auc_from_df()
            rock.get_df()
            self.rock_dict[score_name] = rock

    def combine_dfs(self):
        _ = [rock.get_df() for rock in self.rock_dict.values()]
        poses_dfs = [rock.auc_poses_df for rock in self.rock_dict.values()]
        cmpds_dfs = [rock.auc_cmpds_df for rock in self.rock_dict.values()]
        self.poses_df = pd.concat(poses_dfs)
        self.cmpds_df = pd.concat(cmpds_dfs)

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

    def plot_poses_auc(self):
        fig = ex.line(
            self.poses_df,
            x="False_Positive",
            y="True_Positive",
            color="Score_Type",
            hover_data=["Value"],
        )
        fig.update_layout(height=600, width=600, title="ROC of all POSES")
        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
        )
        fig.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, xref="x", yref="y")

        return fig

    def plot_precision_recall(self):
        fig = ex.line(
            self.poses_df,
            x="True_Positive",
            y="Precision",
            color="Score_Type",
            hover_data=["Value"],
        )
        fig.update_layout(height=600, width=600, title="ROC of all POSES")
        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1,
        )
        return fig

    def get_compound_results_df(self):
        total_poses = self.df.groupby("Compound_ID")["RMSD"].count()
        RMSDs = self.df.groupby("Compound_ID")[["RMSD"]].apply(
            lambda x: x[x <= 2].agg(["count", "min"])
        )
        n_good_poses = RMSDs.xs("count", level=1)["RMSD"]
        min_RMSD = RMSDs.xs("min", level=1)["RMSD"]
        perc_good_poses = n_good_poses / total_poses
        min_posit_R = self.df.groupby("Compound_ID")["POSIT_R"].min()
        cmpd_df = pd.DataFrame(
            {
                "N_Poses": total_poses,
                "N_Good_Poses": n_good_poses,
                "Perc_Good_Poses": perc_good_poses,
                "Min_RMSD": min_RMSD,
                "Min_POSIT_R": min_posit_R,
            }
        )
        cmpd_df["Compound_ID"] = cmpd_df.index
        self.cmpd_df = cmpd_df.sort_values("Perc_Good_Poses")
