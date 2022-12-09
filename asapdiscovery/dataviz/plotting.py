# TODO: Do we need to add plotly to our environment yaml?
import plotly.express as px
from asapdiscovery.dataviz.utils import filter_df_by_two_columns


def plot_poses_auc(poses_df):
    fig = px.line(
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
    fig = px.line(
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


def contour_plot(
    df,
    xaxis_column_name,
    yaxis_column_name,
    xaxis_type,
    yaxis_type,
    x_range,
    y_range,
):
    """
    Make a contour plot of two variables that are each columns in a dataframe
    Values for column names and range sliders.

    Parameters
    ----------
    df
    xaxis_column_name
    yaxis_column_name
    xaxis_type
    yaxis_type
    x_range
    y_range

    Returns
    -------

    """
    filtered = filter_df_by_two_columns(
        df, xaxis_column_name, yaxis_column_name, x_range, y_range
    )

    fig = px.density_contour(
        filtered,
        x=xaxis_column_name,
        y=yaxis_column_name,
        marginal_x="histogram",
        marginal_y="histogram",
    )
    fig.update_traces(
        contours_coloring="heatmap",
        selector=dict(type="histogram2dcontour"),
        colorscale="Peach",
    )
    fig.update_xaxes(
        title=xaxis_column_name,
        type="linear" if xaxis_type == "Linear" else "log",
    )

    fig.update_yaxes(
        title=yaxis_column_name,
        type="linear" if yaxis_type == "Linear" else "log",
    )

    fig.update_layout(
        margin={"l": 40, "b": 40, "t": 40, "r": 40}, hovermode="closest"
    )

    return fig


def scatter_plot(
    df,
    xaxis_column_name,
    yaxis_column_name,
    xaxis_type,
    yaxis_type,
    x_range,
    y_range,
    color_column,
):
    filtered = filter_df_by_two_columns(
        df, xaxis_column_name, yaxis_column_name, x_range, y_range
    )

    fig = px.scatter(
        filtered,
        x=xaxis_column_name,
        y=yaxis_column_name,
        hover_data=["Complex_ID", "Compound_ID", "Structure_Source"],
        color=color_column,
        color_continuous_scale="dense",
    )

    fig.update_xaxes(
        title=xaxis_column_name,
        type="linear" if xaxis_type == "Linear" else "log",
    )

    fig.update_yaxes(
        title=yaxis_column_name,
        type="linear" if yaxis_type == "Linear" else "log",
    )

    fig.update_layout(
        margin={"l": 40, "b": 40, "t": 40, "r": 40}, hovermode="closest"
    )

    return fig
