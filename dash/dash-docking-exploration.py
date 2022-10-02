import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px

app = Dash(__name__)

df = pd.read_csv(
    "/Volumes/Rohirrim/local_test/mers_hallucination_hybrid/posit_hybrid_no_relax/by_structure.csv"
)
tidy_df = df.melt(id_vars="Structure_Source")

app.layout = html.Div(
    [
        html.H4("Interactive bar chart"),
        dcc.Graph(id="per-structure-bar-chart"),
        html.P("Filter by Resolution:"),
        dcc.RangeSlider(
            id="range-slider",
            min=0,
            max=3.5,
            step=0.1,
            marks={0: "0", 1: "1", 3.5: "3.5"},
            value=[0.5, 3.5],
        ),
    ]
)


@app.callback(
    Output("per-structure-bar-chart", "figure"), Input("range-slider", "value")
)
def update_bar_chart(slider_range):
    low, high = slider_range
    mask = (df["Resolution"] > low) & (df["Resolution"] < high)
    filtered_df = df[mask]
    fig = px.bar(
        filtered_df.sort_values("RMSD_Good"),
        x="RMSD_Good",
        y="Structure_Source",
        height=800,
        width=800,
    )
    return fig


app.run_server(port=9000, host="localhost", debug=True)
