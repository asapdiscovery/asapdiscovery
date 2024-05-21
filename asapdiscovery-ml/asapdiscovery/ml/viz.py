import pandas
import seaborn as sns


def plot_split_losses(
    pred_tracker_dict, out_fn=None, splits=["train", "val", "test"], loss_label="Loss"
):
    """
    Plot overall losses per split by training epoch.

    Parameters
    ----------
    pred_tracker_dict : dict[str, TrainingPredictionTracker]
        Dict mapping labels to pred trackers
    out_fn : Path, optional
        Path to save plot to
    splits : list[str], default=["train", "val", "test"]
        Which splits to actually plot
    loss_label : str, default="Loss"
        What to label the y-axis of the plot
    """
    # Build overall DF
    all_dfs = []
    for lab, pred_tracker in pred_tracker_dict.items():
        df = pred_tracker.to_plot_df(agg_compounds=True, agg_losses=True)
        df["label"] = lab
        all_dfs.append(df)

    all_dfs = pandas.concat(all_dfs, ignore_index=True)

    # Subset
    all_dfs = all_dfs.loc[all_dfs["split"].isin(splits), :]

    # Figure out styles
    if len(pred_tracker_dict) > 1:
        # More than one different experiment, so use color for experiment and style
        #  for split
        hue = "label"
        hue_order = list(pred_tracker_dict.keys())
        if len(splits) > 1:
            style = "split"
            style_order = splits
        else:
            style = None
            style_order = None
    else:
        hue = "split"
        hue_order = splits
        style = None
        style_order = None

    # Make plot
    # fig = plt.figure(figsize=(7, 5))
    fg = sns.relplot(
        all_dfs,
        x="epoch",
        y="loss",
        hue=hue,
        style=style,
        hue_order=hue_order,
        style_order=style_order,
        kind="line",
        aspect=1.5,
    )

    # Set axes
    fg.set_axis_labels("Training Epoch", loss_label)

    if out_fn:
        fg.savefig(out_fn, bbox_inches="tight", dpi=200)

    return fg
