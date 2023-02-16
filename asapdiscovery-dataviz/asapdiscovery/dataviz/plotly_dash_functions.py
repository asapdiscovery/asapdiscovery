from dash import Dash, dash_table, dcc, html


def get_dash_app():
    """
    Create basic dash app and styles
    Returns
    -------
    app, styles
    """
    app = Dash(__name__)
    styles = {"pre": {"border": "thin lightgrey solid", "overflowX": "scroll"}}
    return app, styles


def get_basic_plot(id):
    """
    Create basic dash plot

    Parameters
    ----------
    id: name of plot reference in dash syntax

    Returns
    -------

    """
    return html.Div(
        [
            dcc.Graph(
                id=id,
            )
        ],
        style={
            "display": "inline-block",
            "padding": "0 20",
            # "float": "right",
        },
    )


def get_heading(id, text):
    """

    Parameters
    ----------
    id: name of plot reference in dash syntax
    text: text to use in header

    Returns
    -------

    """
    return html.Div(
        [
            html.H4(text),
            # dcc.Graph(id=id),
        ],
        style={"width": "49%", "display": "inline-block"},
    )


def get_color_variable(variable_list, id="crossfilter-color"):
    """
    Add dropdown menu that can be used to select the variable to use to color by

    Parameters
    ----------
    variable_list: list of possible variables in dropdown menu
    id: name of plot reference in dash syntax

    Returns
    -------

    """
    return html.Div(
        [
            html.H4("Color"),
            dcc.Dropdown(
                variable_list,  ## the whole list
                variable_list[0],  ## the default
                id=id,
            ),
        ],
        style={
            "width": "98%",
            "display": "inline-block",
            # "float": "left",
        },
    )


def get_dash_table(id, variable_dict, columns, filter_action="native"):
    """
    Create interactive table.

    Parameters
    ----------
    id: name of plot reference in dash syntax
    variable_dict: dictionary representing the dataframe
    columns: columns of the table
    filter_action: "native" means that each column can be used to manually filter.
        "custom" can be used instead to program in a filtering function for the table

    Returns
    -------

    """
    return dash_table.DataTable(
        variable_dict,
        id=id,
        columns=columns,
        style_table={
            "width": "50%",
            "float": "center",
            "display": "inline-block",
            "padding": "0 20",
            # "float": "right",
        },
        filter_action=filter_action,
        sort_action="native",
        filter_query="",
    )


def get_filters(
    variable_list,
    default_x=None,
    default_y=None,
    xrange=[-100, 100],
    yrange=[-100, 100],
):
    # TODO: split this up into each individual component
    """
    Create filters for subsectioning plotted data

    Parameters
    ----------
    variable_list

    Returns
    -------

    """
    if not default_x:
        default_x = variable_list[0]
    if not default_y:
        default_y = variable_list[-1]

    return html.Div(
        [
            html.Div(
                [
                    html.H4("X-Axis"),
                    dcc.Dropdown(
                        variable_list,  ## the whole list
                        default_x,  ## the default
                        id="crossfilter-xaxis-column",
                    ),
                    html.P("Filter X-Axis:"),
                    dcc.RangeSlider(
                        id="x-axis-slider",
                        min=xrange[0],
                        max=xrange[1],
                        value=xrange,
                    ),
                    dcc.RadioItems(
                        ["Linear", "Log"],
                        "Linear",
                        id="crossfilter-xaxis-type",
                        labelStyle={
                            "display": "inline-block",
                            "marginTop": "5px",
                        },
                    ),
                ],
                style={
                    "width": "49%",
                    "display": "inline-block",
                },
            ),
            html.Div(
                [
                    html.H4("Y-Axis"),
                    dcc.Dropdown(
                        variable_list,  ## the whole list
                        default_y,  ## the default
                        id="crossfilter-yaxis-column",
                    ),
                    html.P("Filter Y-Axis:"),
                    dcc.RangeSlider(
                        id="y-axis-slider",
                        min=yrange[0],
                        max=yrange[1],
                        value=yrange,
                    ),
                    dcc.RadioItems(
                        ["Linear", "Log"],
                        "Linear",
                        id="crossfilter-yaxis-type",
                        labelStyle={
                            "display": "inline-block",
                            "marginTop": "5px",
                        },
                    ),
                ],
                style={
                    "width": "49%",
                    "display": "inline-block",
                },
            ),
        ],
        style={
            "padding": "10px 5px",
        },
    )
