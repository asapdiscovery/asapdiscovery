import logging
import os.path
from openeye import oechem
import numpy as np
import pandas
import pydantic
import re
import rdkit.Chem as Chem

from asapdiscovery.data.schema import (
    ExperimentalCompoundData,
    ExperimentalCompoundDataUpdate,
    CrystalCompoundData,
    EnantiomerPair,
    EnantiomerPairList,
)


def download_file(url: str, path: str):
    """
    Download a file and save it locally.
    Copied from kinoml.utils

    Parameters
    ----------
    url: str
        URL for downloading data.
    path: str
        Path to save downloaded data.

    Returns
    -------
    : bool
        True if successful, else False.
    """
    import requests

    response = requests.get(url)
    if response.status_code == 200:
        with open(path, "wb") as write_file:
            write_file.write(response.content)
        return True

    return False


def get_remark_str(chains, oligomeric_state):
    allowed_states = ["monomer", "dimer"]
    if oligomeric_state == "monomer":
        REMARK350_STRING = f"""\
REMARK 300 SEE REMARK 350 FOR THE AUTHOR PROVIDED AND/OR PROGRAM
REMARK 350
REMARK 350 COORDINATES FOR A COMPLETE MULTIMER REPRESENTING THE KNOWN
REMARK 350 BIOLOGICALLY SIGNIFICANT OLIGOMERIZATION STATE OF THE
REMARK 350 MOLECULE CAN BE GENERATED BY APPLYING BIOMT TRANSFORMATIONS
REMARK 350 GIVEN BELOW.  BOTH NON-CRYSTALLOGRAPHIC AND
REMARK 350 CRYSTALLOGRAPHIC OPERATIONS ARE GIVEN.
REMARK 350
REMARK 350 BIOMOLECULE: 1
REMARK 350 AUTHOR DETERMINED BIOLOGICAL UNIT: DIMERIC
REMARK 350 SOFTWARE DETERMINED QUATERNARY STRUCTURE: DIMERIC
REMARK 350 SOFTWARE USED: PISA
REMARK 350 TOTAL BURIED SURFACE AREA: 4170 ANGSTROM**2
REMARK 350 SURFACE AREA OF THE COMPLEX: 25430 ANGSTROM**2
REMARK 350 CHANGE IN SOLVENT FREE ENERGY: -3.0 KCAL/MOL
REMARK 350 APPLY THE FOLLOWING TO CHAINS: {", ".join(chains)}
REMARK 350   BIOMT1   1  1.000000  0.000000  0.000000        0.00000
REMARK 350   BIOMT2   1  0.000000  1.000000  0.000000        0.00000
REMARK 350   BIOMT3   1  0.000000  0.000000  1.000000        0.00000
REMARK 350   BIOMT1   2 -1.000000  0.000000  0.000000        0.00000
REMARK 350   BIOMT2   2  0.000000  1.000000  0.000000        0.00000
REMARK 350   BIOMT3   2  0.000000  0.000000 -1.000000        0.00000
"""
    elif oligomeric_state == "dimer":
        REMARK350_STRING = f"""\
REMARK 300 SEE REMARK 350 FOR THE AUTHOR PROVIDED AND/OR PROGRAM
REMARK 350
REMARK 350 COORDINATES FOR A COMPLETE MULTIMER REPRESENTING THE KNOWN
REMARK 350 BIOLOGICALLY SIGNIFICANT OLIGOMERIZATION STATE OF THE
REMARK 350 MOLECULE CAN BE GENERATED BY APPLYING BIOMT TRANSFORMATIONS
REMARK 350 GIVEN BELOW.  BOTH NON-CRYSTALLOGRAPHIC AND
REMARK 350 CRYSTALLOGRAPHIC OPERATIONS ARE GIVEN.
REMARK 350
REMARK 350 BIOMOLECULE: 1
REMARK 350 AUTHOR DETERMINED BIOLOGICAL UNIT: DIMERIC
REMARK 350 SOFTWARE DETERMINED QUATERNARY STRUCTURE: DIMERIC
REMARK 350 SOFTWARE USED: PISA
REMARK 350 TOTAL BURIED SURFACE AREA: 1680 ANGSTROM**2
REMARK 350 SURFACE AREA OF THE COMPLEX: 23750 ANGSTROM**2
REMARK 350 CHANGE IN SOLVENT FREE ENERGY: -13.0 KCAL/MOL
REMARK 350 APPLY THE FOLLOWING TO CHAINS: {", ".join(chains)}
REMARK 350   BIOMT1   1  1.000000  0.000000  0.000000        0.00000
REMARK 350   BIOMT2   1  0.000000  1.000000  0.000000        0.00000
REMARK 350   BIOMT3   1  0.000000  0.000000  1.000000        0.00000
"""
    elif oligomeric_state not in allowed_states:
        raise NotImplementedError(
            f"get_remark_str not implemented for {oligomeric_state}. "
            f"Try one of: '{' ,'.join(allowed_states)}'"
        )
    return REMARK350_STRING


def edit_pdb_file(
    pdb_in,
    seqres_str=None,
    dbref_str=None,
    edit_remark350=False,
    oligomeric_state=None,
    chains=None,
    pdb_out=None,
):
    """
    Add SARS-CoV2 MPRO residue sequence to PDB header.

    Parameters
    ----------
    pdb_in : str
        Input PDB file.
    seqres_str : str, optional
        String containing SEQRES card, including newlines.
    dbref_str : str, optional
        String containing DBREF card, including newlines.
    pdb_out : str, optional
        Output PDB file. If not given, appends _seqres to the input file.
    """
    ## TODO: replace DBREF string as well

    pdbfile_lines = [line for line in open(pdb_in, "r") if "UNK" not in line]
    pdbfile_lines = [line for line in pdbfile_lines if "LINK" not in line]

    ## Fix bad CL atom names
    pdbfile_lines = [re.sub("CL", "Cl", l) for l in pdbfile_lines]
    if seqres_str:
        pdbfile_lines = [line for line in pdbfile_lines if not "SEQRES" in line]

        pdbfile_lines = [
            line.rstrip() + "\n" for line in seqres_str.split("\n")
        ] + pdbfile_lines

    if edit_remark350:
        pdbfile_lines = [
            line for line in pdbfile_lines if not "REMARK 350" in line
        ]
        if chains and oligomeric_state:
            remark_str = get_remark_str(chains, oligomeric_state)
            pdbfile_lines = [
                line.rstrip() + "\n" for line in remark_str.split("\n")
            ] + pdbfile_lines
    pdbfile_contents = "".join(pdbfile_lines)

    if pdb_out is None:
        pdb_out = f"{pdb_in[:-4]}_seqres.pdb"
    with open(pdb_out, "w") as fp:
        fp.write(pdbfile_contents)

    print(f"Wrote {pdb_out}", flush=True)


def seqres_to_res_list(seqres_str):
    """
    https://www.wwpdb.org/documentation/file-format-content/format33/sect3.html#SEQRES
    Parameters
    ----------
    SEQRES_str

    Returns
    -------

    """
    ## Grab the sequence from the sequence str
    ## use chain ID column
    seqres_chain_column = 11
    seq_lines = [
        line[19:]
        for line in seqres_str.split("\n")
        if len(line) > 0
        if line[seqres_chain_column] == "A"
    ]
    seq_str = " ".join(seq_lines)
    res_list = seq_str.split(" ")
    return res_list


def cdd_to_schema(cdd_csv, out_json=None, out_csv=None):
    """
    Convert a CDD-downloaded and filtered CSV file into a JSON file containing
    an ExperimentalCompoundDataUpdate. CSV file should be the result of the
    filter_molecules_dataframe function and must contain the following headers:
        * name
        * smiles
        * achiral
        * racemic
        * pIC50
        * pIC50_stderr
        * pIC50_95ci_lower
        * pIC50_95ci_upper
        * pIC50_range

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
    ExperimentalCompoundDataUpdate
        The parsed ExperimentalCompoundDataUpdate.
    """

    ## Load and remove any straggling compounds w/o SMILES data
    df = pandas.read_csv(cdd_csv)

    ## Check that all required columns are present
    reqd_cols = [
        "name",
        "smiles",
        "achiral",
        "racemic",
        "semiquant",
        "pIC50",
        "pIC50_stderr",
        "pIC50_95ci_lower",
        "pIC50_95ci_upper",
        "pIC50_range",
    ]
    missing_cols = [c for c in reqd_cols if c not in df.columns]
    if len(missing_cols) > 0:
        raise ValueError(
            (
                f"Required columns not present in CSV file: {missing_cols}. "
                "Please use `filter_molecules_dataframe` to properly populate "
                "the dataframe."
            )
        )

    ## Make extra sure nothing snuck by
    idx = df["smiles"].isna()
    logging.debug(f"Removing {idx.sum()} entries with no SMILES", flush=True)
    df = df.loc[~idx, :]

    ## Fill standard error for semi-qunatitative data with the mean of others
    df.loc[df["semiquant"], "pIC50_stderr"] = df.loc[
        ~df["semiquant"], "pIC50_stderr"
    ].mean()

    ## For now just keep the first measure for each compound_id (should be the
    ##  only one if `keep_best_per_mol` was set when running
    ##  `filter_molecules_dataframe`.)
    compounds = []
    seen_compounds = {}
    for i, (_, c) in enumerate(df.iterrows()):
        compound_id = c["name"]
        ## Replace long dash unicode character with regular - sign (only
        ##  one compound like this I think)
        if "\u2212" in compound_id:
            print(
                f"Replacing unicode character with - in",
                compound_id,
                flush=True,
            )
            compound_id = re.sub("\u2212", "-", compound_id)
        if compound_id in seen_compounds:
            ## If there are no NaN values, don't need to fix
            if not seen_compounds[compound_id]:
                continue

        smiles = c["smiles"]
        experimental_data = {
            "pIC50": c["pIC50"],
            "pIC50_range": c["pIC50_range"],
            "pIC50_stderr": c["pIC50_stderr"],
        }
        ## Add delta G values if present
        if "exp_binding_affinity_kcal_mol" in c:
            experimental_data.update(
                {
                    "dG": c["exp_binding_affinity_kcal_mol"],
                    "dG_stderr": c["exp_binding_affinity_kcal_mol_stderr"],
                }
            )

        ## Keep track of if there are any NaN values
        try:
            seen_compounds[compound_id] = np.isnan(
                list(experimental_data.values())
            ).any()
        except TypeError as e:

            seen_compounds[compound_id] = True

        try:
            compounds.append(
                ExperimentalCompoundData(
                    compound_id=compound_id,
                    smiles=smiles,
                    racemic=c["racemic"],
                    achiral=c["achiral"],
                    absolute_stereochemistry_enantiomerically_pure=(
                        not c["racemic"]
                    ),
                    relative_stereochemistry_enantiomerically_pure=(
                        not c["racemic"]
                    ),
                    experimental_data=experimental_data,
                )
            )
        except pydantic.error_wrappers.ValidationError as e:
            print(
                "Error converting this row to ExperimentalCompoundData object:",
                c,
                flush=True,
            )
            raise e

    compounds = ExperimentalCompoundDataUpdate(compounds=compounds)

    if out_json:
        with open(out_json, "w") as fp:
            fp.write(compounds.json())
        print(f"Wrote {out_json}", flush=True)
    if out_csv:
        out_cols = [
            "name",
            "smiles",
            "pIC50",
            "pIC50_range",
            "pIC50_95ci_lower",
            "pIC50_95ci_upper",
            "pIC50_stderr",
        ]
        df[out_cols].to_csv(out_csv)
        print(f"Wrote {out_csv}", flush=True)

    return compounds


def cdd_to_schema_pair(cdd_csv, out_json=None, out_csv=None):
    """
    Convert a CDD-downloaded and filtered CSV file into a JSON file containing
    an EnantiomerPairList. CSV file should be the result of the
    filter_molecules_dataframe function and must contain the following headers:
        * name
        * smiles
        * achiral
        * racemic
        * pIC50
        * pIC50_stderr
        * pIC50_95ci_lower
        * pIC50_95ci_upper
        * pIC50_range

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

    ## Check that all required columns are present
    reqd_cols = [
        "name",
        "smiles",
        "achiral",
        "racemic",
        "semiquant",
        "pIC50",
        "pIC50_stderr",
        "pIC50_95ci_lower",
        "pIC50_95ci_upper",
        "pIC50_range",
    ]
    missing_cols = [c for c in reqd_cols if c not in df.columns]
    if len(missing_cols) > 0:
        raise ValueError(
            (
                f"Required columns not present in CSV file: {missing_cols}. "
                "Please use `filter_molecules_dataframe` to properly populate "
                "the dataframe."
            )
        )

    ## Make extra sure nothing snuck by
    idx = df["smiles"].isna()
    logging.debug(f"Removing {idx.sum()} entries with no SMILES", flush=True)
    df = df.loc[~idx, :]

    ## Fill standard error for semi-qunatitative data with the mean of others
    df.loc[df["semiquant"], "pIC50_stderr"] = df.loc[
        ~df["semiquant"], "pIC50_stderr"
    ].mean()

    ## Remove stereochemistry tags and get canonical SMILES values (to help
    ##  group stereoisomers)
    smi_nostereo = [CanonSmiles(s, useChiral=False) for s in df["smiles"]]
    df["smiles_nostereo"] = smi_nostereo

    ## Sort by non-stereo SMILES to put the enantiomer pairs together
    df = df.sort_values("smiles_nostereo")

    enant_pairs = []
    ## Loop through the enantiomer pairs and rank them
    for ep in df.groupby("smiles_nostereo"):
        ## Make sure there aren't any singletons
        if ep[1].shape[0] != 2:
            print(f"{ep[1].shape[0]} mols for {ep[0]}", flush=True)
            continue

        p = []
        ## Sort by pIC50 value, higher to lower
        ep = ep[1].sort_values("pIC50", ascending=False)
        for _, c in ep.iterrows():
            compound_id = c["name"]
            ## Replace long dash unicode character with regular - sign (only
            ##  one compound like this I think)
            if "\u2212" in compound_id:
                print(
                    f"Replacing unicode character with - in",
                    compound_id,
                    flush=True,
                )
                compound_id = re.sub("\u2212", "-", compound_id)
            smiles = c["smiles"]
            experimental_data = {
                "pIC50": c["pIC50"],
                "pIC50_range": c["pIC50_range"],
                "pIC50_stderr": c["pIC50_stderr"],
            }
            ## Add delta G values if present
            if "exp_binding_affinity_kcal_mol" in c:
                experimental_data.update(
                    {
                        "dG": c["exp_binding_affinity_kcal_mol"],
                        "dG_stderr": c["exp_binding_affinity_kcal_mol_stderr"],
                    }
                )

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
            "name",
            "smiles",
            "smiles_nostereo",
            "pIC50",
            "pIC50_range",
            "pIC50_95ci_lower",
            "pIC50_95ci_upper",
            "pIC50_stderr",
        ]
        df[out_cols].to_csv(out_csv)
        print(f"Wrote {out_csv}", flush=True)

    return ep_list


def get_achiral_molecules(mol_df):
    """
    Remove chiral molecules.

    Parameters
    ----------
    mol_df : pandas.DataFrame
        DataFrame containing compound information

    Returns
    -------
    pandas.DataFrame
        DataFrame containing compound information for all achiral molecules

    """
    from rdkit.Chem import FindMolChiralCenters, MolFromSmiles

    ## Check whether a SMILES is chiral or not
    check_achiral = (
        lambda smi: len(
            FindMolChiralCenters(
                MolFromSmiles(smi),
                includeUnassigned=True,
                includeCIP=False,
                useLegacyImplementation=False,
            )
        )
        == 0
    )
    ## Check each molecule, first looking at suspected_SMILES, then
    ##  shipment_SMILES if not present
    achiral_idx = []
    for _, r in mol_df.iterrows():
        if ("suspected_SMILES" in r) and (
            not pandas.isna(r["suspected_SMILES"])
        ):
            achiral_idx.append(check_achiral(r["suspected_SMILES"]))
        elif ("shipment_SMILES" in r) and (
            not pandas.isna(r["shipment_SMILES"])
        ):
            achiral_idx.append(check_achiral(r["shipment_SMILES"]))
        else:
            raise ValueError(f'No SMILES found for {r["Canonical PostEra ID"]}')

    return mol_df.loc[achiral_idx, :]


def strip_smiles_salts(smiles):
    """
    Strip salts from a SMILES string.

    Parameters
    ----------
    smiles : str
        SMILES containig salt(s) to remove

    Returns
    -------
    str
        Salt-free SMILES
    """

    oemol = oechem.OEGraphMol()
    oechem.OESmilesToMol(oemol, smiles)
    oechem.OEDeleteEverythingExceptTheFirstLargestComponent(oemol)

    return oechem.OEMolToSmiles(oemol)


def filter_molecules_dataframe(
    mol_df,
    smiles_fieldname="suspected_SMILES",
    retain_achiral=False,
    retain_racemic=False,
    retain_enantiopure=False,
    retain_semiquantitative_data=False,
    keep_best_per_mol=True,
    assay_name="ProteaseAssay_Fluorescence_Dose-Response_Weizmann",
    dG_T=298.0,
):
    """
    Filter a dataframe of molecules to retain those specified.

    For example, to filter a DF of molecules so that it only contains achiral
    molecules while allowing for measurements that are semiquantitative:
    `mol_df = filter_molecules_dataframe(
        mol_df,
        retain_achiral=True,
        retain_semiquantitative_data=True
    )`

    Parameters
    ----------
    mol_df : pandas.DataFrame
        DataFrame containing compound information
    smiles_fieldname : str, default="suspected_SMILES"
        Field name to use for reference SMILES
    retain_achiral : bool, default=False
        If True, retain achiral measurements
    retain_racemic : bool, default=False
        If True, retain racemic measurements
    retain_enantiopure : bool, default=False
        If True, retain chirally resolved measurements
    retain_semiquantitative_data : bool, default=False
        If True, retain semiquantitative data (data outside assay dynamic range)
    keep_best_per_mol : bool, default=True
        Keep only the best measurement for each molecule (first sorting by
        curve class and then 95% CI pIC50 width)
    assay_name : str, default="ProteaseAssay_Fluorescence_Dose-Response_Weizmann"
        Name of the assay of interest
    dG_T : float, default=298.0
        Temperature in Kelvin for converting pIC50 values to delta G values

    Returns
    -------
    pandas.DataFrame
        DataFrame containing compound information for all filtered molecules
    """
    import logging
    import numpy as np
    from rdkit.Chem import FindMolChiralCenters, MolFromSmiles

    # Define functions to evaluate whether molecule is achiral, racemic, or resolved
    is_achiral = (
        lambda smi: len(
            FindMolChiralCenters(
                MolFromSmiles(smi),
                includeUnassigned=True,
                includeCIP=False,
                useLegacyImplementation=False,
            )
        )
        == 0
    )
    is_racemic = (
        lambda smi: (
            len(
                FindMolChiralCenters(
                    MolFromSmiles(smi),
                    includeUnassigned=True,
                    includeCIP=False,
                    useLegacyImplementation=False,
                )
            )
            - len(
                FindMolChiralCenters(
                    MolFromSmiles(smi),
                    includeUnassigned=False,
                    includeCIP=False,
                    useLegacyImplementation=False,
                )
            )
        )
        > 0
    )
    is_enantiopure = lambda smi: (not is_achiral(smi)) and (not is_racemic(smi))

    def is_semiquant(ic50):
        try:
            _ = float(ic50)
            return False
        except ValueError as e:
            return True

    logging.debug(f"  dataframe contains {mol_df.shape[0]} entries")

    ## Drop any rows with no SMILES (need the copy to make pandas happy)
    # Get rid of any molecules that snuck through without SMILES field specified
    mol_df = mol_df.dropna(subset=smiles_fieldname).copy()
    logging.debug(
        (
            f"  dataframe contains {mol_df.shape[0]} entries after removing "
            f"molecules with unspecified {smiles_fieldname} field"
        )
    )

    ## Add new columns so we can keep the original names
    logging.debug("Stripping salts")
    mol_df.loc[:, "smiles"] = (
        mol_df.loc[:, smiles_fieldname].astype(str).apply(strip_smiles_salts)
    )
    mol_df.loc[:, "name"] = mol_df.loc[:, "Canonical PostEra ID"]

    # Convert CXSMILES to SMILES by removing extra info
    mol_df.loc[:, "smiles"] = [
        s.strip("|").split()[0] for s in mol_df.loc[:, "smiles"]
    ]

    logging.debug("Filtering molecules dataframe")
    ## Determine which molecules will be retained and add corresponding labels
    ##  to the data frame
    achiral_label = [is_achiral(smiles) for smiles in mol_df["smiles"]]
    racemic_label = [is_racemic(smiles) for smiles in mol_df["smiles"]]
    enantiopure_label = [is_enantiopure(smiles) for smiles in mol_df["smiles"]]
    semiquant_label = [
        is_semiquant(ic50) for ic50 in mol_df[f"{assay_name}: IC50 (µM)"]
    ]
    keep_idx = [
        (retain_achiral and achiral_label[i])
        or (retain_racemic and racemic_label[i])
        or (retain_enantiopure and enantiopure_label[i])
        or (retain_semiquantitative_data and semiquant_label[i])
        for i in range(mol_df.shape[0])
    ]

    mol_df["achiral"] = achiral_label
    mol_df["racemic"] = racemic_label
    mol_df["enantiopure"] = enantiopure_label
    mol_df["semiquant"] = semiquant_label

    mol_df = mol_df.loc[keep_idx, :]
    logging.debug(
        f"  dataframe contains {mol_df.shape[0]} entries after filtering"
    )

    # Compute pIC50s and uncertainties from 95% CIs
    pIC50_series = []
    pIC50_stderr_series = []
    pIC50_range_series = []
    pIC50_lower_series = []
    pIC50_upper_series = []
    for _, row in mol_df.iterrows():
        try:
            IC50 = float(row[f"{assay_name}: IC50 (µM)"]) * 1e-6  # molar
            IC50_lower = (
                float(row[f"{assay_name}: IC50 CI (Lower) (µM)"]) * 1e-6
            )  # molar
            IC50_upper = (
                float(row[f"{assay_name}: IC50 CI (Upper) (µM)"]) * 1e-6
            )  # molar

            pIC50 = -np.log10(IC50)
            pIC50_lower = -np.log10(IC50_upper)
            pIC50_upper = -np.log10(IC50_lower)
            pIC50_stderr = (
                np.abs(pIC50_upper - pIC50_lower) / 4.0
            )  # assume normal distribution

            # Render into string with appropriate sig figs
            try:
                import sigfig

                pIC50, pIC50_stderr = sigfig.round(
                    pIC50, uncertainty=pIC50_stderr, sep=tuple
                )  # strings
            except ModuleNotFoundError:
                ## Just round to 4 digits if sigfig pacakge not present
                pIC50 = str(round(pIC50, 4))
                pIC50_stderr = str(round(pIC50_stderr, 4))

        except ValueError:
            IC50 = row[f"{assay_name}: IC50 (µM)"]
            # Could not convert to string because value was semiquantitative
            if (
                row[f"{assay_name}: IC50 (µM)"]
                == "(IC50 could not be calculated)"
            ):
                pIC50 = "nan"
            elif ">" in IC50:
                pIC50 = "< 4.0"  # lower limit of detection
            elif "<" in IC50:
                pIC50 = "> 7.3"  # upper limit of detection
            else:
                pIC50 = "nan"

            # Keep pIC50 string
            # Use default pIC50 error
            # print(row)
            ## Set as high number so sorting works but still puts this at end
            pIC50_stderr = 100
            pIC50_lower = np.nan
            pIC50_upper = np.nan

        pIC50_series.append(float(pIC50.strip("<> ")))
        pIC50_stderr_series.append(float(pIC50_stderr))
        ## Add label indicating whether pIC50 values were out of the assay range
        pIC50_range_series.append(
            -1 if "<" in pIC50 else (1 if ">" in pIC50 else 0)
        )
        pIC50_lower_series.append(pIC50_lower)
        pIC50_upper_series.append(pIC50_upper)

    mol_df["pIC50"] = pIC50_series
    mol_df["pIC50_stderr"] = pIC50_stderr_series
    mol_df["pIC50_range"] = pIC50_range_series
    mol_df["pIC50_95ci_lower"] = pIC50_lower_series
    mol_df["pIC50_95ci_upper"] = pIC50_upper_series

    ## Compute binding affinity in kcal/mol
    # use R = .001987 kcal/K/mol
    deltaG = lambda pIC50: -0.001987 * dG_T * np.log(10.0) * float(pIC50)
    mol_df["exp_binding_affinity_kcal_mol"] = [
        deltaG(pIC50) if not np.isnan(pIC50) else np.nan
        for pIC50 in mol_df["pIC50"]
    ]
    mol_df["exp_binding_affinity_kcal_mol_stderr"] = [
        abs(deltaG(pIC50_stderr)) if not np.isnan(pIC50_stderr) else np.nan
        for pIC50_stderr in mol_df["pIC50_stderr"]
    ]
    mol_df["exp_binding_affinity_kcal_mol_95ci_lower"] = [
        deltaG(pIC50_lower) if not np.isnan(pIC50_lower) else np.nan
        for pIC50_lower in mol_df["pIC50_95ci_lower"]
    ]
    mol_df["exp_binding_affinity_kcal_mol_95ci_upper"] = [
        deltaG(pIC50_upper) if not np.isnan(pIC50_upper) else np.nan
        for pIC50_upper in mol_df["pIC50_95ci_upper"]
    ]

    ## Keep only the best measurement for each molecule
    if keep_best_per_mol:
        for mol_name, g in mol_df.groupby("name"):
            g.sort_values(
                by=[f"{assay_name}: Curve class", "pIC50_stderr"],
                inplace=True,
                ascending=True,
            )
        mol_df = mol_df.groupby("name", as_index=False).first()

    return mol_df


def get_sdf_fn_from_dataset(
    dataset: str,
    fragalysis_dir,
):
    fn = os.path.join(fragalysis_dir, f"{dataset}_0A/{dataset}_0A.sdf")
    if not os.path.exists(fn):
        print(f"File {fn} not found...")
        fn = None  ## not sure what behaviour this should have
    return fn


def parse_experimental_compound_data(exp_fn: str, json_fn: str):
    ## Load experimental data and trim columns
    exp_df = pandas.read_csv(exp_fn)
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
            experimental_data={
                "IC50": r["IC50"],
                "IC50_range": r["IC50_range"],
            },
        )
        for _, r in exp_df.iterrows()
    ]

    ## Dump JSON file
    with open(json_fn, "w") as fp:
        fp.write(
            ExperimentalCompoundDataUpdate(compounds=exp_data_compounds).json()
        )


def parse_fragalysis_data(frag_fn, x_dir, cmpd_ids=None, o_dir=False):
    ## Load in csv
    sars2_structures = pandas.read_csv(frag_fn).fillna("")

    if cmpd_ids is not None:
        ## Filter fragalysis dataset by the compounds we want to test
        sars2_filtered = sars2_structures[
            sars2_structures["Compound ID"].isin(cmpd_ids)
        ]
    else:
        sars2_filtered = sars2_structures

    if o_dir:
        mols_wo_sars2_xtal = sars2_filtered[sars2_filtered["Dataset"].isna()][
            ["Compound ID", "SMILES", "Dataset"]
        ]
        mols_w_sars2_xtal = sars2_filtered[~sars2_filtered["Dataset"].isna()][
            ["Compound ID", "SMILES", "Dataset"]
        ]

        ## Use utils function to get sdf file from dataset
        mols_w_sars2_xtal["SDF"] = mols_w_sars2_xtal["Dataset"].apply(
            get_sdf_fn_from_dataset, fragalysis_dir=x_dir
        )

        ## Save csv files for each dataset
        mols_wo_sars2_xtal.to_csv(
            os.path.join(o_dir, "mers_ligands_without_SARS2_structures.csv"),
            index=False,
        )

        mols_w_sars2_xtal.to_csv(
            os.path.join(o_dir, "mers_ligands_with_SARS2_structures.csv"),
            index=False,
        )

    ## Construct sars_xtal list
    sars_xtals = {}
    for data in sars2_filtered.to_dict("index").values():
        cmpd_id = data["Compound ID"]
        dataset = data["Dataset"]
        if len(dataset) > 0:
            ## TODO: is this the behaviour we want? this will build an empty object if there isn't a dataset
            if not sars_xtals.get(cmpd_id) or "-P" in dataset:
                sars_xtals[cmpd_id] = CrystalCompoundData(
                    smiles=data["SMILES"],
                    compound_id=cmpd_id,
                    dataset=dataset,
                    sdf_fn=get_sdf_fn_from_dataset(dataset, x_dir),
                )
        else:
            sars_xtals[cmpd_id] = CrystalCompoundData()

    return sars_xtals


def get_compound_id_xtal_dicts(sars_xtals):
    """
    Get a pair of dictionaries that map between crystal structures and compound
    ids.

    Parameters
    ----------
    sars_xtals : Iter[CrystalCompoundData]
        Iterable of CrystalCompoundData objects from fragalysis.

    Returns
    -------
    Dict[str: List[str]]
        Dict mapping compound_id to list of crystal structure ids.
    Dict[str: str]
        Dict mapping crystal structure id to compound_id.
    """
    compound_to_xtals = {}
    xtal_to_compound = {}
    for ccd in sars_xtals:
        compound_id = ccd.compound_id
        dataset = ccd.dataset
        try:
            compound_to_xtals[compound_id].append(dataset)
        except KeyError:
            compound_to_xtals[compound_id] = [dataset]

        xtal_to_compound[dataset] = compound_id

    return (compound_to_xtals, xtal_to_compound)


def trim_small_chains(input_mol, cutoff_len=10):
    """
    Remove short chains from a protein molecule object. The goal is to get rid
    of any peptide ligands that were mistakenly collected by OESplitMolComplex.

    Parameters
    ----------
    input_mol : oechem.OEGraphMol
        OEGraphMol object containing the protein to trim
    cutoff_len : int, default=10
        The cutoff length for peptide chains (chains must have more than this
        many residues to be included)

    Returns
    -------
    oechem.OEGraphMol
        Trimmed molecule
    """
    ## Copy the molecule
    mol_copy = input_mol.CreateCopy()

    ## Remove chains from mol_copy that are too short (possibly a better way of
    ##  doing this with OpenEye functions)
    ## Get number of residues per chain
    chain_len_dict = {}
    hv = oechem.OEHierView(mol_copy)
    for chain in hv.GetChains():
        chain_id = chain.GetChainID()
        for frag in chain.GetFragments():
            frag_len = len(list(frag.GetResidues()))
            try:
                chain_len_dict[chain_id] += frag_len
            except KeyError:
                chain_len_dict[chain_id] = frag_len

    ## Remove short chains atom by atom
    for a in mol_copy.GetAtoms():
        chain_id = oechem.OEAtomGetResidue(a).GetExtChainID()
        if (chain_id not in chain_len_dict) or (
            chain_len_dict[chain_id] <= cutoff_len
        ):
            mol_copy.DeleteAtom(a)

    return mol_copy


def get_ligand_rmsd_openeye(ref: oechem.OEMolBase, mobile: oechem.OEMolBase):
    return oechem.OERMSD(ref, mobile)


def get_ligand_RMSD_mdtraj(ref_fn, mobile_fn):
    import mdtraj as md

    ref = md.load_pdb(ref_fn)
    mobile = md.load_pdb(mobile_fn)

    ref_idx = ref.topology.select("resname LIG and not type H and chainid 1")
    mobile_idx = mobile.topology.select("resname LIG and not type H")
    print(ref_idx)
    print(mobile_idx)

    ref_lig = ref.atom_slice(ref_idx)
    mobile_lig = mobile.atom_slice(mobile_idx)
    print(ref_lig)
    print(mobile_lig)

    rmsd_array = md.rmsd(ref_lig, mobile_lig, precentered=True)
    per_res_rmsd = rmsd_array[0] / ref_lig.n_atoms
    #
    rmsd_array2 = md.rmsd(
        ref, mobile, atom_indices=mobile_idx, ref_atom_indices=ref_idx
    )
    print(rmsd_array, per_res_rmsd, rmsd_array2)


def filter_docking_inputs(
    smarts_queries="../../data/smarts_queries.csv",
    docking_inputs=None,
    drop_commented_smarts_strings=True,
    verbose=True,
):
    """
    Filter an input file of compound SMILES by SMARTS filters using OEchem matching.

    Parameters
    ----------
    smarts_queries : str
        Path to file containing SMARTS entries to filter by (comma-separated).
    docking_inputs : dict(Compound_ID: smiles)
        Dict containing SMILES entries and ligand names to filter using smarts_queries.
    drop_commented_smarts_strings : bool
        How to handle first-character hashtags ('commented') on SMARTS entries. False
        ignores hashtags so all SMARTS filters are always applied; if True (default), the code ignores
        SMARTS filters that are commented (hashtagged).
    verbose : bool
        Whether or not to print a message stating the number of compounds filtered.

    Returns
    ----------
    filtered_docking_inputs : list
        List containing compounds remaining after applied filter(s), equal format
        to docking_inputs.

    """
    query_smarts = pandas.read_csv(smarts_queries, names=["smarts", "id"])[
        "smarts"
    ].values

    if drop_commented_smarts_strings:
        # only keep SMARTS queries that are not commented.
        query_smarts = [q for q in query_smarts if not q[0] == "#"]
    else:
        # some of the SMARTS queries are commented - use these anyway.
        query_smarts = [q if not q[0] == "#" else q[1:] for q in query_smarts]

    num_input_cpds = 0  # initiate counter for verbose setting.
    filtered_docking_inputs = []
    for cpd, smiles in docking_inputs.items():
        num_input_cpds += 1
        # read input cpd into OE.
        mol = oechem.OEGraphMol()
        oechem.OESmilesToMol(mol, smiles)

        # now loop over queried SMARTS patterns, flag input compound if hit.
        for query in query_smarts:
            # create a substructure search object.
            ss = oechem.OESubSearch(query)
            oechem.OEPrepareSearch(mol, ss)

            # compare this query to the reference mol.
            if ss.SingleMatch(mol):
                # if match is found we can stop querying and output the cpd.
                filtered_docking_inputs.append(cpd)
                break

    if verbose:
        print(
            f"Retained {len(filtered_docking_inputs) / num_input_cpds * 100:.2f}% of compounds after "
            + f"filtering ({len(query_smarts)} SMARTS filter(s); {num_input_cpds}-->"
            + f"{len(filtered_docking_inputs)})."
        )

    # return the filtered list.
    return filtered_docking_inputs


def load_exp_from_sdf(fn):
    """
    Build a list of ExperimentalCompoundData objects from an SDF file.
    Everything other than `compound_id` and `smiles` will be left as default.
    TODO: Use rdkit functions to assign stereochemistry (if 3D SDF file)

    Parameters
    ----------
    fn : str
        SDF file name.

    Returns
    -------
    List[ExperimentalCompoundData]
        List of ExperimentalCompoundData objects parsed from SDF file.
    """
    ## Open SDF file and load all SMILES
    suppl = Chem.rdmolfiles.SDMolSupplier(fn)
    exp_data_compounds = [
        ExperimentalCompoundData(
            compound_id=str(i), smiles=Chem.MolToSmiles(mol)
        )
        for i, mol in enumerate(suppl)
    ]

    return exp_data_compounds


if __name__ == "__main__":
    filter_docking_inputs()
