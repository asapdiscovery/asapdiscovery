def filter_df_by_two_columns(
    df,
    xaxis_column_name,
    yaxis_column_name,
    x_range=None,
    y_range=None,
):
    """
    Simple function for filtering a dataframe by the values of particular columns

    Parameters
    ----------
    df
    xaxis_column_name: str
    yaxis_column_name: str
    x_range: [min, max]
    y_range: [min, max

    Returns
    -------

    """
    if not x_range:
        x_range = (df[xaxis_column_name].min(), df[xaxis_column_name].max())
    if not y_range:
        y_range = (df[yaxis_column_name].min(), df[yaxis_column_name].max())

    dff = df[
        (df[xaxis_column_name] > x_range[0])
        & (df[xaxis_column_name] < x_range[1])
        & (df[yaxis_column_name] > y_range[0])
        & (df[yaxis_column_name] < y_range[1])
    ]
    return dff
