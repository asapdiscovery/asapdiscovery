#!/bin/env python
"""
Filter CDD export for main medicinal chemistry series
https://app.collaborativedrug.com/vaults/5549/searches/11042338-xOBBXlC_s3dSQsHW3UaO2Q#search_results
Header:
Molecule Name,Canonical PostEra ID,suspected_SMILES,ProteaseAssay_Fluorescence_Dose-Response_Weizmann: IC50 (µM),ProteaseAssay_Fluorescence_Dose-Response_Weizmann: IC50 CI (Lower) (µM),ProteaseAssay_Fluorescence_Dose-Response_Weizmann: IC50 CI (Upper) (µM),ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Hill slope,ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Curve class

From: https://github.com/choderalab/perses/blob/main/examples/moonshot-mainseries/molecules/filter-cdd-export.py
"""
import sys
import numpy as np

cdd_csv_filename = "CDD CSV Export - 2023-02-20 00h18m18s.csv"
clean_output_csv_filename = "fullseries_clean.csv"
bulky_output_csv_filename = "fullseries_bulky.csv"

# Load in data to pandas dataframe
import pandas as pd

bulky_df = pd.read_csv(cdd_csv_filename, dtype=str)
print(f"{len(bulky_df)} records read")

# keep only intended columns
bulky_df = bulky_df[
    [
        "Molecule Name",
        "Canonical PostEra ID",
        "suspected_SMILES",
        "ProteaseAssay_Fluorescence_Dose-Response_Weizmann: IC50 (µM)",
        "ProteaseAssay_Fluorescence_Dose-Response_Weizmann: IC50 CI (Lower) (µM)",
        "ProteaseAssay_Fluorescence_Dose-Response_Weizmann: IC50 CI (Upper) (µM)",
        "ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Hill slope",
        "ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Curve class",
        "SMILES",
    ]
]

# Drop NaNs
print(f"Dropping NaNs...")
cleaned_df = bulky_df.dropna(axis=0, how="any", thresh=None, subset=None, inplace=False)
print(f"{len(cleaned_df)} records remain")


# Drop any with ambiguity in suspected SMILES (which will have spaces)
cleaned_df = cleaned_df[
    cleaned_df["suspected_SMILES"].apply(
        lambda x: True if len(x.split()) == 1 else False
    )
]


# Some operations to get the bulky version of the output CSV:
# for the bulky version: instead of dropping, use regular SMILES as suspected_SMILES. Also replace NaNs with regular SMILES.
for idx, row in bulky_df[bulky_df["suspected_SMILES"].isna()].iterrows():
    # set this row's suspected SMILES to regular SMILES
    bulky_df.loc[idx]["suspected_SMILES"] = row["SMILES"]
# same procedure for missing postera ID tags.
for idx, row in bulky_df[bulky_df["Canonical PostEra ID"].isna()].iterrows():
    # set this row's ID tag to indexed.
    bulky_df.loc[idx]["Canonical PostEra ID"] = f"inserted_ID_{idx}"
# for the bulky version: instead of dropping, remove the stereo annotation between pipes.
stripped_smiles = [smi.split(" ")[0] for smi in bulky_df["suspected_SMILES"].values]
bulky_df["suspected_SMILES"] = stripped_smiles
# for bulky, fill in non-quantitative rows. These are ["< 99.5", NaN, NaN] for [IC50, upper, lower].
for idx, row in bulky_df[
    bulky_df[
        "ProteaseAssay_Fluorescence_Dose-Response_Weizmann: IC50 CI (Lower) (µM)"
    ].isna()
].iterrows():
    # set this row's artificial values.
    bulky_df.loc[idx][
        "ProteaseAssay_Fluorescence_Dose-Response_Weizmann: IC50 (µM)"
    ] = 100
    bulky_df.loc[idx][
        "ProteaseAssay_Fluorescence_Dose-Response_Weizmann: IC50 CI (Upper) (µM)"
    ] = 99
    bulky_df.loc[idx][
        "ProteaseAssay_Fluorescence_Dose-Response_Weizmann: IC50 CI (Lower) (µM)"
    ] = 101

# finally drop rows where "(IC50 CI (x) could not be calculated; confidence interval too large)".
bulky_df[
    "ProteaseAssay_Fluorescence_Dose-Response_Weizmann: IC50 CI (Upper) (µM)"
] = pd.to_numeric(
    bulky_df["ProteaseAssay_Fluorescence_Dose-Response_Weizmann: IC50 CI (Upper) (µM)"],
    errors="coerce",
)
bulky_df.dropna(
    subset=["ProteaseAssay_Fluorescence_Dose-Response_Weizmann: IC50 CI (Upper) (µM)"],
    axis=0,
    inplace=True,
)
##

# Compute 95%CI width
import numpy as np


def pIC50(IC50_series):
    return -np.log10(IC50_series.astype(float) * 1e-6)


def DeltaG(pIC50):
    kT = 0.593  # kcal/mol for 298 K (25C)
    return -kT * np.log(10.0) * pIC50


for df, output_path in zip(
    [cleaned_df, bulky_df], [clean_output_csv_filename, bulky_output_csv_filename]
):
    # Rename
    df["Title"] = df["Canonical PostEra ID"]
    df["SMILES"] = df["suspected_SMILES"]

    df["pIC50_95%CI_LOW"] = pIC50(
        df["ProteaseAssay_Fluorescence_Dose-Response_Weizmann: IC50 CI (Upper) (µM)"]
    )
    df["pIC50_95%CI_HIGH"] = pIC50(
        df["ProteaseAssay_Fluorescence_Dose-Response_Weizmann: IC50 CI (Lower) (µM)"]
    )
    df["95% pIC50 width"] = abs(
        pIC50(
            df[
                "ProteaseAssay_Fluorescence_Dose-Response_Weizmann: IC50 CI (Upper) (µM)"
            ]
        )
        - pIC50(
            df[
                "ProteaseAssay_Fluorescence_Dose-Response_Weizmann: IC50 CI (Lower) (µM)"
            ]
        )
    )
    df["pIC50"] = pIC50(
        df["ProteaseAssay_Fluorescence_Dose-Response_Weizmann: IC50 (µM)"]
    )
    df["dpIC50"] = df["95% pIC50 width"] / 4.0  # estimate of standard error

    df["EXP_BINDING_AFFINITY_IN_KCAL_PER_MOL"] = DeltaG(df["pIC50"])
    df["EXP_BINDING_AFFINITY_IN_KCAL_PER_MOL_STDERR"] = abs(DeltaG(df["dpIC50"]))
    df["EXP_BINDING_AFFINITY_IN_KCAL_PER_MOL_95%CI_LOW"] = DeltaG(
        df["pIC50_95%CI_HIGH"]
    )
    df["EXP_BINDING_AFFINITY_IN_KCAL_PER_MOL_95%CI_HIGH"] = DeltaG(
        df["pIC50_95%CI_LOW"]
    )

    # Filter molecules
    print(
        "Keeping best measurements for each molecules, sorting by curve class and then 95% pIC50 width"
    )
    for molecule_name, molecule_group in df.groupby("Canonical PostEra ID", sort=False):
        molecule_group.sort_values(
            by=[
                "ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Curve class",
                "95% pIC50 width",
            ],
            inplace=True,
            ascending=True,
        )

    print("Resulting measurements")
    df = df.groupby("Canonical PostEra ID", sort=False).first()
    print(df)
    print(f"{len(df)} records remain")

    # Write molecules
    df.to_csv(
        output_path,
        columns=[
            "SMILES",
            "Title",
            "pIC50",
            "dpIC50",
            "EXP_BINDING_AFFINITY_IN_KCAL_PER_MOL",
            "EXP_BINDING_AFFINITY_IN_KCAL_PER_MOL_STDERR",
            "EXP_BINDING_AFFINITY_IN_KCAL_PER_MOL_95%CI_LOW",
            "EXP_BINDING_AFFINITY_IN_KCAL_PER_MOL_95%CI_HIGH",
        ],
        index=False,
    )
