# TODO: Do we need to add plotly to our environment yaml?
import plotly.express as ex


def plot_poses_auc(poses_df):
    fig = ex.line(
        poses_df,
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


def plot_precision_recall(poses_df):
    fig = ex.line(
        poses_df,
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
