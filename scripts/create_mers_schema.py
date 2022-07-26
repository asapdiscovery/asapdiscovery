import argparse
import json
import os
import pandas
import sys

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../")
from covid_moonshot_ml.schema import (
    ExperimentalCompoundData,
    ExperimentalCompoundDataUpdate,
)

################################################################################

## Load experimental data and trim columns
exp_data_fn = "COVID_Moonshot_Takeda_panCorona_enzyme_measurements.csv"
exp_df = pandas.read_csv(exp_data_fn)
exp_cols = [
    "External ID",
    "SMILES",
    "Pan-coronavirus_enzymatic_Takeda: IC50 MERS Mpro (μM)",
]
exp_df = exp_df.loc[:, exp_cols].copy()
exp_df.columns = ["External ID", "SMILES", "IC50"]

## Convert semi-quantitative values into floats and trim any NaNs
exp_df = exp_df.loc[~exp_df["IC50"].isna(), :]
ic50_range = [
    -1 if "<" in c else (1 if ">" in c else 0) for c in exp_df["IC50"]
]
ic50_vals = [float(c.strip("<> ")) for c in exp_df["IC50"]]
exp_df.loc[:, "IC50"] = ic50_vals
exp_df["IC50_range"] = ic50_range

## Convert to schema
exp_data_compounds = [
    ExperimentalCompoundData(
        compound_id=r["External ID"],
        smiles=r["SMILES"],
        experimental_data={"IC50": r["IC50"], "IC50_range": r["IC50_range"]},
    )
    for _, r in exp_df.iterrows()
]

## Dump JSON file
schema_fn = "takeda_mers_exp.json"
with open(schema_fn, "w") as fp:
    fp.write(
        ExperimentalCompoundDataUpdate(compounds=exp_data_compounds).json()
    )
