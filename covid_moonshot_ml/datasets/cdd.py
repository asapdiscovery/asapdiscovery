import numpy as np
import pandas

from covid_moonshot_ml.datasets.utils import get_achiral_molecules
from covid_moonshot_ml.schema import (
    ExperimentalCompoundData,
    ExperimentalCompoundDataUpdate,
    EnantiomerPair,
    EnantiomerPairList,
)


def cdd_to_schema(cdd_csv, out_json=None, out_csv=None, achiral=False):
    """
    Convert a CDD-downloaded CSV file into a JSON file containing an
    ExperimentalCompoundDataUpdate. CSV file must contain the following headers:
        * suspected_SMILES
        * Canonical PostEra ID
        * ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Avg pIC50

    Parameters
    ----------
    cdd_csv : str
        CSV file downloaded from CDD.
    out_json : str, optional
        JSON file to save to.
    out_csv : str, optional
        CSV file to save to.
    achiral : bool, default=False
        Only keep achiral molecules

    Returns
    -------
    ExperimentalCompoundDataUpdate
        The parsed ExperimentalCompoundDataUpdate.
    """

    ## Load and remove any straggling compounds w/o SMILES data
    df = pandas.read_csv(cdd_csv)
    df = df.loc[~df["suspected_SMILES"].isna(), :]

    ## Filter out chiral molecules if requested
    achiral_df = get_achiral_molecules(df)
    if achiral:
        df = achiral_df.copy()
    achiral_label = [
        compound_id in achiral_df["Canonical PostEra ID"].values
        for compound_id in df["Canonical PostEra ID"]
    ]

    ## Get rid of the </> signs, since we really only need the values to sort
    ##  enantiomer pairs
    pic50_key = "ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Avg pIC50"
    df = df.loc[~df[pic50_key].isna(), :]
    pic50_range = [
        -1 if "<" in c else (1 if ">" in c else 0) for c in df[pic50_key]
    ]
    pic50_vals = [float(c.strip("<> ")) for c in df[pic50_key]]
    df["pIC50"] = pic50_vals
    df["pIC50_range"] = pic50_range
    semiquant = df["pIC50_range"].astype(bool)

    ci_lower_key = (
        "ProteaseAssay_Fluorescence_Dose-Response_Weizmann: IC50 "
        "CI (Lower) (µM)"
    )
    ci_upper_key = (
        "ProteaseAssay_Fluorescence_Dose-Response_Weizmann: IC50 "
        "CI (Upper) (µM)"
    )
    ## Convert IC50 vals to pIC50 and calculate standard error for  95% CI
    pic50_stderr = []
    for _, (ci_lower, ci_upper) in df[[ci_lower_key, ci_upper_key]].iterrows():
        if pandas.isna(ci_lower) or pandas.isna(ci_upper):
            pic50_stderr.append(np.nan)
        else:
            ## First convert bounds from IC50 (uM) to pIC50
            pic50_ci_upper = -np.log10(ci_upper * 10e-6)
            pic50_ci_lower = -np.log10(ci_lower * 10e-6)
            ## Assume size of 95% CI == 4*sigma
            pic50_stderr.append((pic50_ci_lower - pic50_ci_upper) / 4)
    df["pIC50_stderr"] = pic50_stderr
    ## Fill standard error for semi-qunatitative data with the mean of others
    df.loc[semiquant, "pIC50_stderr"] = df.loc[
        ~semiquant, "pIC50_stderr"
    ].mean()

    compounds = []
    for i, (_, c) in enumerate(df.iterrows()):
        compound_id = c["Canonical PostEra ID"]
        smiles = c["suspected_SMILES"]
        experimental_data = {
            "pIC50": c["pIC50"],
            "pIC50_range": c["pIC50_range"],
            "pIC50_stderr": c["pIC50_stderr"],
        }

        compounds.append(
            ExperimentalCompoundData(
                compound_id=compound_id,
                smiles=smiles,
                racemic=False,
                achiral=achiral_label[i],
                absolute_stereochemistry_enantiomerically_pure=True,
                relative_stereochemistry_enantiomerically_pure=True,
                experimental_data=experimental_data,
            )
        )
    compounds = ExperimentalCompoundDataUpdate(compounds=compounds)

    if out_json:
        with open(out_json, "w") as fp:
            fp.write(compounds.json())
        print(f"Wrote {out_json}", flush=True)
    if out_csv:
        out_cols = [
            "Canonical PostEra ID",
            "suspected_SMILES",
            "pIC50",
            "pIC50_range",
            ci_lower_key,
            ci_upper_key,
            "pIC50_stderr",
        ]
        df[out_cols].to_csv(out_csv)
        print(f"Wrote {out_csv}", flush=True)

    return compounds


def cdd_to_schema_pair(cdd_csv, out_json=None, out_csv=None):
    """
    Convert a CDD-downloaded CSV file into a JSON file containing an
    EnantiomerPairList. CSV file must contain the following headers:
        * suspected_SMILES
        * Canonical PostEra ID
        * ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Avg pIC50

    Parameters
    ----------
    cdd_csv : str
        CSV file downloaded from CDD.
    out_json : str, optional
        JSON file to save to.
    out_csv : str, optional
        CSV file to save to.

    Returns
    -------
    EnantiomerPairList
        The parsed EnantiomerPairList.
    """
    from rdkit.Chem import CanonSmiles

    ## Load and remove any straggling compounds w/o SMILES data
    df = pandas.read_csv(cdd_csv)
    df = df.loc[~df["suspected_SMILES"].isna(), :]

    ## Remove stereochemistry tags and get canonical SMILES values (to help
    ##  group stereoisomers)
    smi_nostereo = [
        CanonSmiles(s, useChiral=False) for s in df["suspected_SMILES"]
    ]
    df["suspected_SMILES_nostereo"] = smi_nostereo

    ## Sort by non-stereo SMILES to put the enantiomer pairs together
    df = df.sort_values("suspected_SMILES_nostereo")

    ## Get rid of the </> signs, since we really only need the values to sort
    ##  enantiomer pairs
    pic50_key = "ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Avg pIC50"
    pic50_range = [
        -1 if "<" in c else (1 if ">" in c else 0) for c in df[pic50_key]
    ]
    pic50_vals = [float(c[pic50_key].strip("<> ")) for _, c in df.iterrows()]
    df["pIC50"] = pic50_vals
    df["pIC50_range"] = pic50_range
    semiquant = df["pIC50_range"].astype(bool)

    ci_lower_key = (
        "ProteaseAssay_Fluorescence_Dose-Response_Weizmann: IC50 "
        "CI (Lower) (µM)"
    )
    ci_upper_key = (
        "ProteaseAssay_Fluorescence_Dose-Response_Weizmann: IC50 "
        "CI (Upper) (µM)"
    )
    ## Calculate 95% CI in pIC50 units based on IC50 vals (not sure if the
    ##  difference should be taken before or after taking the -log10)
    pic50_stderr = []
    for _, (ci_lower, ci_upper) in df[[ci_lower_key, ci_upper_key]].iterrows():
        if pandas.isna(ci_lower) or pandas.isna(ci_upper):
            pic50_stderr.append(np.nan)
        else:
            ## First convert bounds from IC50 (uM) to pIC50
            pic50_ci_upper = -np.log10(ci_upper * 10e-6)
            pic50_ci_lower = -np.log10(ci_lower * 10e-6)
            ## Assume size of 95% CI == 4*sigma => calculate variance from stdev
            pic50_stderr.append((pic50_ci_lower - pic50_ci_upper) / 4)
    df["pIC50_stderr"] = pic50_stderr
    ## Fill standard error for semi-qunatitative data with the mean of others
    df.loc[semiquant, "pIC50_stderr"] = df.loc[
        ~semiquant, "pIC50_stderr"
    ].mean()

    enant_pairs = []
    ## Loop through the enantiomer pairs and rank them
    for ep in df.groupby("suspected_SMILES_nostereo"):
        ## Make sure there aren't any singletons
        if ep[1].shape[0] != 2:
            print(f"{ep[1].shape[0]} mols for {ep[0]}", flush=True)
            continue

        p = []
        ## Sort by pIC50 value, higher to lower
        ep = ep[1].sort_values("pIC50", ascending=False)
        for _, c in ep.iterrows():
            compound_id = c["Canonical PostEra ID"]
            smiles = c["suspected_SMILES"]
            experimental_data = {
                "pIC50": c["pIC50"],
                "pIC50_range": c["pIC50_range"],
                "pIC50_stderr": c["pIC50_stderr"],
            }

            p.append(
                ExperimentalCompoundData(
                    compound_id=compound_id,
                    smiles=smiles,
                    racemic=False,
                    achiral=False,
                    absolute_stereochemistry_enantiomerically_pure=True,
                    relative_stereochemistry_enantiomerically_pure=True,
                    experimental_data=experimental_data,
                )
            )

        enant_pairs.append(EnantiomerPair(active=p[0], inactive=p[1]))

    ep_list = EnantiomerPairList(pairs=enant_pairs)

    if out_json:
        with open(out_json, "w") as fp:
            fp.write(ep_list.json())
        print(f"Wrote {out_json}", flush=True)
    if out_csv:
        out_cols = [
            "Canonical PostEra ID",
            "suspected_SMILES",
            "suspected_SMILES_nostereo",
            "pIC50",
            "pIC50_range",
            ci_lower_key,
            ci_upper_key,
            "pIC50_stderr",
        ]
        df[out_cols].to_csv(out_csv)
        print(f"Wrote {out_csv}", flush=True)

    return ep_list
