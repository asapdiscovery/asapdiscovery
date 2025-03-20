import glob
import logging
import os.path
import re
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas
from asapdiscovery.data.backend.openeye import oechem
from asapdiscovery.data.schema.experimental import ExperimentalCompoundData
from asapdiscovery.data.schema.legacy import EnantiomerPair, EnantiomerPairList
from pydantic.v1 import ValidationError

# Not sure if this is the right place for these
# Regex patterns for extracting Mpro dataset ID and Moonshot CDD style compound ID
#  from filenames. This is used eg in building ML datasets. For more details on any of
#  these regexes, you can visit regex101.com and get a full breakdown of what each
#  component does

# This will match any string that follows the original COVID Moonshot naming convention,
#  eg: AAR-POS-5507155c-1
MOONSHOT_CDD_ID_REGEX = r"[A-Z]{3}-[A-Z]{3}-[0-9a-z]+-[0-9]+"
# This will match any string that follows the Fragalysis naming convention for the Mpro
#  structures, eg:  Mpro-P2005_0A
MPRO_ID_REGEX = r"Mpro-.*?_[0-9][A-Z]"
ASAP_ID_REGEX = "ASAP-[0-9]{7}"

# Regex patterns that match chains as well, but only capture the main part. These
#  regexes are used when we want to group files based on their unique identifier (ie the
#  captured group), but still be able to keep track of the full name including the chain

# This will match any string that follows the original COVID Moonshot naming convention
#  and also contains a PDB chain, eg: AAR-POS-5507155c-1_0A
# In this example, it will capture AAR-POS-5507155c-1
MOONSHOT_CDD_ID_REGEX_CAPT = r"([A-Z]{3}-[A-Z]{3}-[a-z0-9]+-[0-9]+)_[0-9][A-Z]"
# This will match any string that follows the Fragalysis naming convention for the Mpro
#  structures, eg:  Mpro-P2005_0A
# In this example, it will capture Mpro-P2005
MPRO_ID_REGEX_CAPT = r"(Mpro-[A-Za-z][0-9]+)_[0-9][A-Z]"


def construct_regex_function(pat, fail_val=None, ret_groups=False):
    """
    Construct a function that searches for the given regex pattern, either returning
    fail_val or raising an error if no match is found.
    The output of the returned function will depend on the value passed for
    ``ret_groups``. If ``True``, then both the overall match and the tuple of captured
    groups will be returned. If ``False``, only one value will be returned. If there is
    a capture group in ``pat``, we assume that's what should be matched and will return
    the first captured group. Otherwise, the full match will be returned.

    Parameters
    ----------
    pat : str
        Regular expression to search for
    fail_val : str, optional
        If a value is passed, this value will be returned from the re searches if a
        match isn't found. If None (default), a ValueError will be raised from the re
        search
    ret_groups : bool, default=False
        If True, return the whole match, as well as any groups that were captured

    Returns
    -------
    regex_func: Callable
        Fucntion that searches for pattern in `pat`
    """

    def regex_func(s):
        import re

        m = re.search(pat, s)
        if m:
            if ret_groups:
                return m.group(), m.groups()
            elif len(m.groups()) > 0:
                # Take capture group to be what we're looking for
                return m.groups()[0]
            else:
                return m.group()
        elif fail_val is not None:
            return fail_val
        else:
            raise ValueError(f"No match found for pattern {pat} in {s}.")

    return regex_func


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
    requests.Response
        HTTP response from the GET attempt
    """
    import requests

    response = requests.get(url)
    if response.status_code == 200:
        with open(path, "wb") as write_file:
            write_file.write(response.content)

    return response


def extract_compounds_from_filenames(fn_list, xtal_pat, compound_pat, fail_val=None):
    """
    Extract a list of (xtal, compound_id) from fn_list.

    Parameters
    ----------
    fn_list : List[str]
        List of filenames
    xtal_pat : Union[str, function]
        Regex pattern or function for extracting crystal structure ID from filename. If
        a function is passed, it is expected to return a single str giving the xtal name
    compound_pat : Union[str, function]
        Regex pattern or function for extracting crystal structure ID from filename. If
        a function is passed, it is expected to return a single str giving the
        compound_id
    fail_val : str, optional
        If a value is passed, this value will be returned from the re searches if a
        match isn't found. If None (default), a ValueError will be raised from the re
        search

    Returns
    -------
    List[Tuple[str, str]]
        List of (xtal, compound_id)
    """
    if callable(xtal_pat):
        # Just use the passed function
        xtal_func = xtal_pat
    else:
        # Construct function for re searching
        xtal_func = construct_regex_function(xtal_pat, fail_val)

    if callable(compound_pat):
        # Just use the passed function
        compound_func = compound_pat
    else:
        # Construct function for re searching
        compound_func = construct_regex_function(compound_pat, fail_val)

    return [(xtal_func(fn), compound_func(fn)) for fn in fn_list]


def seqres_to_res_list(seqres_str):
    """
    https://www.wwpdb.org/documentation/file-format-content/format33/sect3.html#SEQRES
    Parameters
    ----------
    seqres_str

    Returns
    -------

    """
    # Grab the sequence from the sequence str
    # use chain ID column
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
        a list[ExperimentalCompoundData]. CSV file should be the result of the
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
    s
        Returns
        -------
        list[ExperimentalCompoundData]
            The parsed list of ExperimentalCompoundData objects.
    """

    # Load and remove any straggling compounds w/o SMILES data
    df = pandas.read_csv(cdd_csv)

    # Check that all required columns are present
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
            f"Required columns not present in CSV file: {missing_cols}. "
            "Please use `filter_molecules_dataframe` to properly populate "
            "the dataframe."
        )

    # Make extra sure nothing snuck by
    idx = df["smiles"].isna()
    logging.debug(f"Removing {idx.sum()} entries with no SMILES", flush=True)
    df = df.loc[~idx, :]

    # Fill standard error for semi-qunatitative data with the mean of others
    df.loc[df["semiquant"], "pIC50_stderr"] = df.loc[
        ~df["semiquant"], "pIC50_stderr"
    ].mean()

    # For now just keep the first measure for each compound_id (should be the
    #  only one if `keep_best_per_mol` was set when running
    #  `filter_molecules_dataframe`.)
    compounds = []
    seen_compounds = {}
    for i, (_, c) in enumerate(df.iterrows()):
        compound_id = c["name"]
        # Replace long dash unicode character with regular - sign (only
        #  one compound like this I think)
        if "\u2212" in compound_id:
            print(
                "Replacing unicode character with - in",
                compound_id,
                flush=True,
            )
            compound_id = re.sub("\u2212", "-", compound_id)
        if compound_id in seen_compounds:
            # If there are no NaN values, don't need to fix
            if not seen_compounds[compound_id]:
                continue

        smiles = c["smiles"]
        experimental_data = {
            "pIC50": c["pIC50"],
            "pIC50_range": c["pIC50_range"],
            "pIC50_stderr": c["pIC50_stderr"],
        }
        # Add delta G values if present
        if "exp_binding_affinity_kcal_mol" in c:
            experimental_data.update(
                {
                    "dG": c["exp_binding_affinity_kcal_mol"],
                    "dG_stderr": c["exp_binding_affinity_kcal_mol_stderr"],
                    "dG_95ci_lower": c["exp_binding_affinity_kcal_mol_95ci_lower"],
                    "dG_95ci_upper": c["exp_binding_affinity_kcal_mol_95ci_upper"],
                }
            )
        if "exp_binding_affinity_kT" in c:
            experimental_data.update(
                {
                    "dG_kT": c["exp_binding_affinity_kT"],
                    "dG_kT_stderr": c["exp_binding_affinity_kT_stderr"],
                    "dG_kT_95ci_lower": c["exp_binding_affinity_kT_95ci_lower"],
                    "dG_kT_95ci_upper": c["exp_binding_affinity_kT_95ci_upper"],
                }
            )

        # Add date created if present
        if "Batch Created Date" in c.index:
            date_created = pandas.to_datetime(c["Batch Created Date"]).date()
        else:
            date_created = None

        # Keep track of if there are any NaN values
        try:
            seen_compounds[compound_id] = np.isnan(
                list(experimental_data.values())
            ).any()
        except TypeError:
            seen_compounds[compound_id] = True

        try:
            compounds.append(
                ExperimentalCompoundData(
                    compound_id=compound_id,
                    smiles=smiles,
                    racemic=c["racemic"],
                    achiral=c["achiral"],
                    absolute_stereochemistry_enantiomerically_pure=(not c["racemic"]),
                    relative_stereochemistry_enantiomerically_pure=(not c["racemic"]),
                    date_created=date_created,
                    experimental_data=experimental_data,
                )
            )
        except ValidationError as e:
            print(
                "Error converting this row to ExperimentalCompoundData object:",
                c,
                flush=True,
            )
            raise e

    if out_json:
        with open(out_json, "w") as fp:
            fp.write("[" + ", ".join([c.json() for c in compounds]) + "]")
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
        if "Batch Created Date" in df.columns:
            out_cols += ["Batch Created Date"]
        df[out_cols].to_csv(out_csv)
        print(f"Wrote {out_csv}", flush=True)

    return compounds


def cdd_to_schema_v2(
    cdd_csv, target_prop, time_column=None, out_json=None, out_csv=None
):
    """
    Convert a CDD-downloaded and filtered CSV file into a JSON file containing
    a list[ExperimentalCompoundData]. CSV file should be the result of the
    filter_molecules_dataframe function and must contain the following headers:
    * name
    * smiles
    * achiral
    * racemic
    * <target property>

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
    list[ExperimentalCompoundData]
        The parsed list of ExperimentalCompoundData objects.
    """

    # Load and remove any straggling compounds w/o SMILES data
    df = pandas.read_csv(cdd_csv)

    # Check that all required columns are present
    reqd_cols = [
        "name",
        "smiles",
        "achiral",
        "racemic",
        "semiquant",
        target_prop,
    ]
    missing_cols = [c for c in reqd_cols if c not in df.columns]
    if len(missing_cols) > 0:
        raise ValueError(f"Required columns not present in CSV file: {missing_cols}. ")

    # Make extra sure nothing snuck by
    idx = df["smiles"].isna()
    logging.debug(f"Removing {idx.sum()} entries with no SMILES", flush=True)
    df = df.loc[~idx, :]

    # Fill semi-qunatitative data with the mean of others WHAT TO DO FOR regular scalar?
    df.loc[df["semiquant"], target_prop] = df.loc[~df["semiquant"], target_prop].mean()

    # For now just keep the first measure for each compound_id (should be the
    #  only one if `keep_best_per_mol` was set when running
    #  `filter_molecules_dataframe`.)
    compounds = []
    seen_compounds = {}
    for _, c in df.iterrows():
        compound_id = c["name"]
        # take first observation is this right?
        if compound_id in seen_compounds:
            # If there are no NaN values, don't need to fix
            if not seen_compounds[compound_id]:
                continue

        smiles = c["smiles"]
        experimental_data = {}
        experimental_data[target_prop] = c[target_prop]

        # Add date created if present
        if time_column in c.index:
            date_created = pandas.to_datetime(c[time_column]).date()
        else:
            date_created = None

        # Keep track of if there are any NaN values
        try:
            seen_compounds[compound_id] = np.isnan(
                list(experimental_data.values())
            ).any()
        except TypeError:
            seen_compounds[compound_id] = True

        try:
            compounds.append(
                ExperimentalCompoundData(
                    compound_id=compound_id,
                    smiles=smiles,
                    racemic=c["racemic"],
                    achiral=c["achiral"],
                    absolute_stereochemistry_enantiomerically_pure=(not c["racemic"]),
                    relative_stereochemistry_enantiomerically_pure=(not c["racemic"]),
                    date_created=date_created,
                    experimental_data=experimental_data,
                )
            )
        except ValidationError as e:
            print(
                "Error converting this row to ExperimentalCompoundData object:",
                c,
                flush=True,
            )
            raise e

    # needs to be this way for compatibility #TODO: change this
    # should be json.dumps([c.json() for c in compounds])
    if out_json:
        with open(out_json, "w") as fp:
            fp.write("[" + ", ".join([c.json() for c in compounds]) + "]")
        print(f"Wrote {out_json}", flush=True)

    if out_csv:
        # read schema into dataframe
        schema_df = pandas.DataFrame([c.dict() for c in compounds])
        # write to csv
        schema_df.to_csv(out_csv)

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

    # Load and remove any straggling compounds w/o SMILES data
    df = pandas.read_csv(cdd_csv)

    # Check that all required columns are present
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
            f"Required columns not present in CSV file: {missing_cols}. "
            "Please use `filter_molecules_dataframe` to properly populate "
            "the dataframe."
        )

    # Make extra sure nothing snuck by
    idx = df["smiles"].isna()
    logging.debug(f"Removing {idx.sum()} entries with no SMILES", flush=True)
    df = df.loc[~idx, :]

    # Fill standard error for semi-qunatitative data with the mean of others
    df.loc[df["semiquant"], "pIC50_stderr"] = df.loc[
        ~df["semiquant"], "pIC50_stderr"
    ].mean()

    # Remove stereochemistry tags and get canonical SMILES values (to help
    #  group stereoisomers)
    smi_nostereo = [CanonSmiles(s, useChiral=False) for s in df["smiles"]]
    df["smiles_nostereo"] = smi_nostereo

    # Sort by non-stereo SMILES to put the enantiomer pairs together
    df = df.sort_values("smiles_nostereo")

    enant_pairs = []
    # Loop through the enantiomer pairs and rank them
    for ep in df.groupby("smiles_nostereo"):
        # Make sure there aren't any singletons
        if ep[1].shape[0] != 2:
            print(f"{ep[1].shape[0]} mols for {ep[0]}", flush=True)
            continue

        p = []
        # Sort by pIC50 value, higher to lower
        ep = ep[1].sort_values("pIC50", ascending=False)
        for _, c in ep.iterrows():
            compound_id = c["name"]
            # Replace long dash unicode character with regular - sign (only
            #  one compound like this I think)
            if "\u2212" in compound_id:
                print(
                    "Replacing unicode character with - in",
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
            # Add delta G values if present
            if "exp_binding_affinity_kcal_mol" in c:
                experimental_data.update(
                    {
                        "dG": c["exp_binding_affinity_kcal_mol"],
                        "dG_stderr": c["exp_binding_affinity_kcal_mol_stderr"],
                    }
                )

            # Add date created if present
            if "Batch Created Date" in c.index:
                date_created = pandas.to_datetime(c["Batch Created Date"]).date()
            else:
                date_created = None

            p.append(
                ExperimentalCompoundData(
                    compound_id=compound_id,
                    smiles=smiles,
                    racemic=False,
                    achiral=False,
                    absolute_stereochemistry_enantiomerically_pure=True,
                    relative_stereochemistry_enantiomerically_pure=True,
                    date_created=date_created,
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
        if "Batch Created Date" in df.columns:
            out_cols += ["Batch Created Date"]

        df[out_cols].to_csv(out_csv)
        print(f"Wrote {out_csv}", flush=True)

    return ep_list


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
    id_fieldname="Canonical PostEra ID",
    smiles_fieldname="suspected_SMILES",
    assay_name="ProteaseAssay_Fluorescence_Dose-Response_Weizmann",
    retain_achiral=False,
    retain_racemic=False,
    retain_enantiopure=False,
    retain_semiquantitative_data=False,
    retain_invalid=False,
    is_ic50=True,
):
    """
    Filter a dataframe of molecules to retain those specified. Required columns are:
    * `id_fieldname`
    * `smiles_fieldname`
    * "`assay_name`: IC50 (µM)"
    Columns that are added to the dataframe by this function:
    * "name"
    * "smiles"
    * "achiral"
    * "racemic"
    * "enantiopure"
    * "semiquant"

    For example, to filter a DF of molecules so that it only contains achiral
    molecules while allowing for measurements that are semiquantitative:
    `mol_df = filter_molecules_dataframe(mol_df, retain_achiral=True, retain_semiquantitative_data=True)`

    Parameters
    ----------
    mol_df : pandas.DataFrame
        DataFrame containing compound information
    smiles_fieldname : str, default="suspected_SMILES"
        Field name to use for reference SMILES
    assay_name : str, default="ProteaseAssay_Fluorescence_Dose-Response_Weizmann"
        Name of the assay of interest
    retain_achiral : bool, default=False
        If True, retain achiral measurements
    retain_racemic : bool, default=False
        If True, retain racemic measurements
    retain_enantiopure : bool, default=False
        If True, retain chirally resolved measurements
    retain_semiquantitative_data : bool, default=False
        If True, retain semiquantitative data (data outside assay dynamic range)
    retain_invalid : bool, default=False
        If True, retain data with IC50 values that could not be calculated

    Returns
    -------
    pandas.DataFrame
        DataFrame containing compound information for all filtered molecules
    """
    import logging

    from rdkit.Chem import FindMolChiralCenters, MolFromSmiles

    # Define functions to evaluate whether molecule is achiral, racemic, or resolved
    def is_achiral(smi):
        return (
            len(
                FindMolChiralCenters(
                    MolFromSmiles(smi),
                    includeUnassigned=True,
                    includeCIP=False,
                    useLegacyImplementation=False,
                )
            )
            == 0
        )

    def is_racemic(smi):
        return (
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
            > 0
        )

    is_enantiopure = lambda smi: (not is_achiral(smi)) and (  # noqa: E731
        not is_racemic(smi)
    )

    def is_semiquant(val):
        try:
            _ = float(val)
            return False
        except ValueError:
            return True

    def is_invalid(val):
        try:
            _ = float(val)
            return False
        except ValueError:
            if "<" in val or ">" in val:
                return False
            return True

    logging.debug(f"  dataframe contains {mol_df.shape[0]} entries")

    # Drop any rows with no SMILES (need the copy to make pandas happy)
    # Get rid of any molecules that snuck through without SMILES field specified
    mol_df = mol_df.dropna(subset=smiles_fieldname).copy()
    logging.debug(
        f"  dataframe contains {mol_df.shape[0]} entries after removing "
        f"molecules with unspecified {smiles_fieldname} field"
    )

    # Add new columns so we can keep the original names
    logging.debug("Stripping salts")
    mol_df.loc[:, "smiles"] = (
        mol_df.loc[:, smiles_fieldname].astype(str).apply(strip_smiles_salts)
    )
    mol_df.loc[:, "name"] = mol_df.loc[:, id_fieldname]

    # Convert CXSMILES to SMILES by removing extra info
    mol_df.loc[:, "smiles"] = [s.strip("|").split()[0] for s in mol_df.loc[:, "smiles"]]

    logging.debug("Filtering molecules dataframe")
    # Determine which molecules will be retained and add corresponding labels
    #  to the data frame
    achiral_label = [is_achiral(smiles) for smiles in mol_df["smiles"]]
    racemic_label = [is_racemic(smiles) for smiles in mol_df["smiles"]]
    enantiopure_label = [is_enantiopure(smiles) for smiles in mol_df["smiles"]]
    if is_ic50:
        semiquant_label = [
            is_semiquant(ic50) for ic50 in mol_df[f"{assay_name}: IC50 (µM)"]
        ]
        invalid_label = [
            is_invalid(ic50) for ic50 in mol_df[f"{assay_name}: IC50 (µM)"]
        ]
    else:
        semiquant_label = [is_semiquant(val) for val in mol_df[f"{assay_name}"]]
        invalid_label = [is_invalid(val) for val in mol_df[f"{assay_name}"]]
    mol_df["achiral"] = achiral_label
    mol_df["racemic"] = racemic_label
    mol_df["enantiopure"] = enantiopure_label
    mol_df["semiquant"] = semiquant_label
    mol_df["invalid"] = invalid_label

    # Check which molcules to keep
    achiral_keep_idx = np.asarray([retain_achiral and lab for lab in achiral_label])
    racemic_keep_idx = np.asarray([retain_racemic and lab for lab in racemic_label])
    enantiopure_keep_idx = np.asarray(
        [retain_enantiopure and lab for lab in enantiopure_label]
    )
    keep_idx = achiral_keep_idx | racemic_keep_idx | enantiopure_keep_idx

    # If we do want to keep semiquant data, don't need to do any further filtering
    if not retain_semiquantitative_data:
        # Only want to keep non semi-quant data, so negate label first before taking &
        keep_idx &= ~np.asarray(semiquant_label)

    # Same with invalid data
    if not retain_invalid:
        keep_idx &= ~np.asarray(invalid_label)

    mol_df = mol_df.loc[keep_idx, :]
    logging.debug(f"  dataframe contains {mol_df.shape[0]} entries after filtering")

    return mol_df


def parse_fluorescence_data_cdd(
    mol_df,
    keep_best_per_mol=True,
    assay_name="ProteaseAssay_Fluorescence_Dose-Response_Weizmann",
    dG_T=298.0,
    cp_values=None,
    pic50_stderr_filt=10.0,
):
    """
    Filter a dataframe of molecules to retain those specified. Required columns are:
        * "name"
        * "`assay_name`: IC50 (µM)"
        * "`assay_name`: IC50 CI (Lower) (µM)"
        * "`assay_name`: IC50 CI (Upper) (µM)"
        * "`assay_name`: Curve class"
    Columns that are added to the dataframe by this function:
        * "IC50 (M)"
        * "IC50_stderr (M)"
        * "IC50_95ci_lower (M)"
        * "IC50_95ci_upper (M)"
        * "pIC50"
        * "pIC50_stderr"
        * "pIC50_range"
        * "pIC50_95ci_lower"
        * "pIC50_95ci_upper"
        * "exp_binding_affinity_kcal_mol"
        * "exp_binding_affinity_kcal_mol_stderr"
        * "exp_binding_affinity_kcal_mol_95ci_lower"
        * "exp_binding_affinity_kcal_mol_95ci_upper"
        * "exp_binding_affinity_kT"
        * "exp_binding_affinity_kT_stderr"
        * "exp_binding_affinity_kT_95ci_lower"
        * "exp_binding_affinity_kT_95ci_upper"

    Parameters
    ----------
    mol_df : pandas.DataFrame
        DataFrame containing compound information
    keep_best_per_mol : bool, default=True
        Keep only the best measurement for each molecule (first sorting by
        curve class and then 95% CI pIC50 width)
    assay_name : str, default="ProteaseAssay_Fluorescence_Dose-Response_Weizmann"
        Name of the assay of interest
    dG_T : float, default=298.0
        Temperature in Kelvin for converting pIC50 values to delta G values
    cp_values : Tuple[int], optional
        Substrate concentration and Km values for calculating Ki using the
        Cheng-Prussoff equation. These values are assumed to be in the same
        concentration units. If no values are passed for this, pIC50 values
        will be used as an approximation of the Ki
    pic50_stderr_filt : float, default=10.0
        Max allowable standard error in pIC50 units. Overly large errors lead to rounded
        values that don't make sense, so set anything that will cause issues to NaN

    Returns
    -------
    pandas.DataFrame
        DataFrame containing parsed fluorescence and binding affinity values
    """
    import logging

    import numpy as np

    # Create a copy so we don't modify the original
    mol_df = mol_df.copy()

    # Compute pIC50s and uncertainties from 95% CIs
    IC50_series = []
    IC50_stderr_series = []
    IC50_lower_series = []
    IC50_upper_series = []
    pIC50_series = []
    pIC50_stderr_series = []
    pIC50_range_series = []
    pIC50_lower_series = []
    pIC50_upper_series = []
    pic50_filt_compounds = []
    for _, row in mol_df.iterrows():
        try:
            IC50 = float(row[f"{assay_name}: IC50 (µM)"])
            pIC50 = -np.log10(IC50 * 1e-6)
            pIC50_range = 0
        except ValueError:
            IC50 = row[f"{assay_name}: IC50 (µM)"]
            # Could not convert to string because value was semiquantitative
            if IC50 == "(IC50 could not be calculated)":
                IC50 = "nan"
                pIC50 = "nan"
                pIC50_range = 0
            elif ">" in IC50 or "<" in IC50:
                # Label indicating whether pIC50 values were out of the assay range
                # Signs are flipped bc we are assigning based on IC50 but the value
                #  applies to pIC50
                pIC50_range = -1 if ">" in IC50 else 1
                IC50 = float(IC50.strip("<> "))
                pIC50 = round(-np.log10(IC50 * 1e-6), 2)
            else:
                IC50 = "nan"
                pIC50 = "nan"
                pIC50_range = 0

        try:
            IC50_lower = float(row[f"{assay_name}: IC50 CI (Lower) (µM)"])
            IC50_upper = float(row[f"{assay_name}: IC50 CI (Upper) (µM)"])
            if np.isnan(IC50_lower) or np.isnan(IC50_upper):
                raise ValueError
            IC50_stderr = (
                np.abs(IC50_upper - IC50_lower) / 4.0
            )  # assume normal distribution

            pIC50_lower = -np.log10(IC50_upper * 1e-6)
            pIC50_upper = -np.log10(IC50_lower * 1e-6)
            pIC50_stderr = (
                np.abs(pIC50_upper - pIC50_lower) / 4.0
            )  # assume normal distribution
        except ValueError:
            # Keep pIC50 string
            # Use default pIC50 error
            # print(row)
            # Set as high number so sorting works but still puts this at end
            IC50_stderr = "100"
            IC50_lower = np.nan
            IC50_upper = np.nan
            pIC50_stderr = "100"
            pIC50_lower = np.nan
            pIC50_upper = np.nan

        if (
            isinstance(IC50, float)
            and (pIC50_range == 0)
            and isinstance(IC50_stderr, float)
        ):
            # Check error for filtering
            if pIC50_stderr > pic50_stderr_filt:
                # Set everything to NaN
                IC50 = np.nan
                pIC50 = np.nan
                IC50_stderr = np.nan
                IC50_lower = np.nan
                IC50_upper = np.nan
                pIC50_stderr = np.nan
                pIC50_lower = np.nan
                pIC50_upper = np.nan

                # Store the compound to log later
                pic50_filt_compounds.append(row["name"])
            else:
                # Have numbers for IC50 and stderr so can do rounding
                try:
                    import sigfig

                    # BUG: rounding here with large error bars can cause the values to be clipped
                    # to 0 or 10, we should just drop these. See #1234
                    IC50, IC50_stderr = sigfig.round(
                        IC50, uncertainty=IC50_stderr, sep=tuple, output_type=str
                    )  # strings
                    pIC50, pIC50_stderr = sigfig.round(
                        pIC50, uncertainty=pIC50_stderr, sep=tuple, output_type=str
                    )  # strings
                except ModuleNotFoundError:
                    # Don't round
                    pass

        IC50_series.append(float(IC50) * 1e-6)
        IC50_stderr_series.append(float(IC50_stderr) * 1e-6)
        IC50_lower_series.append(IC50_lower * 1e-6)
        IC50_upper_series.append(IC50_upper * 1e-6)
        pIC50_series.append(float(pIC50))
        pIC50_stderr_series.append(float(pIC50_stderr))
        # Add label indicating whether pIC50 values were out of the assay range
        pIC50_range_series.append(pIC50_range)
        pIC50_lower_series.append(pIC50_lower)
        pIC50_upper_series.append(pIC50_upper)

    # Let the user know we found some invalid values
    if len(pic50_filt_compounds) > 0:
        logging.debug(
            (
                "These compounds had standard errors outside the set range and were "
                f"filtered to NaNs: {pic50_filt_compounds}"
            ),
        )

    mol_df["IC50 (M)"] = IC50_series
    mol_df["IC50_stderr (M)"] = IC50_stderr_series
    mol_df["IC50_95ci_lower (M)"] = IC50_lower_series
    mol_df["IC50_95ci_upper (M)"] = IC50_upper_series
    mol_df["pIC50"] = pIC50_series
    mol_df["pIC50_stderr"] = pIC50_stderr_series
    mol_df["pIC50_range"] = pIC50_range_series
    mol_df["pIC50_95ci_lower"] = pIC50_lower_series
    mol_df["pIC50_95ci_upper"] = pIC50_upper_series

    # Compute binding affinity in kcal/mol
    try:
        from simtk.unit import MOLAR_GAS_CONSTANT_R as R_const
        from simtk.unit import kelvin as K
        from simtk.unit import kilocalorie as kcal
        from simtk.unit import mole as mol

        R = R_const.in_units_of(kcal / mol / K)._value
    except ModuleNotFoundError:
        # use R = .001987 kcal/K/mol
        R = 0.001987
        logging.debug("simtk package not found, using R value of", R)

    # Calculate Ki using Cheng-Prussoff
    if cp_values:
        logging.debug("Using Cheng-Prussoff equation for delta G calculations")

        # IC50 in M
        def deltaG(IC50):
            return R * dG_T * np.log(IC50 / (1 + cp_values[0] / cp_values[1]))

        # dG in implicit kT units
        def deltaG_kT(IC50):
            return np.log(IC50 / (1 + cp_values[0] / cp_values[1]))

        mol_df["exp_binding_affinity_kcal_mol"] = [
            deltaG(IC50) if not np.isnan(IC50) else np.nan
            for IC50 in mol_df["IC50 (M)"]
        ]
        mol_df["exp_binding_affinity_kT"] = [
            deltaG_kT(IC50) if not np.isnan(IC50) else np.nan
            for IC50 in mol_df["IC50 (M)"]
        ]
        mol_df["exp_binding_affinity_kcal_mol_95ci_lower"] = [
            deltaG(IC50_lower) if not np.isnan(IC50_lower) else np.nan
            for IC50_lower in mol_df["IC50_95ci_lower (M)"]
        ]
        mol_df["exp_binding_affinity_kT_95ci_lower"] = [
            deltaG_kT(IC50_lower) if not np.isnan(IC50_lower) else np.nan
            for IC50_lower in mol_df["IC50_95ci_lower (M)"]
        ]
        mol_df["exp_binding_affinity_kcal_mol_95ci_upper"] = [
            deltaG(IC50_upper) if not np.isnan(IC50_upper) else np.nan
            for IC50_upper in mol_df["IC50_95ci_upper (M)"]
        ]
        mol_df["exp_binding_affinity_kT_95ci_upper"] = [
            deltaG_kT(IC50_upper) if not np.isnan(IC50_upper) else np.nan
            for IC50_upper in mol_df["IC50_95ci_upper (M)"]
        ]
    else:
        logging.debug("Using pIC50 values for delta G calculations")

        def deltaG(pIC50):
            return -R * dG_T * np.log(10.0) * pIC50

        # dG in implicit kT units
        def deltaG_kT(pIC50):
            return np.log(10.0) * pIC50

        mol_df["exp_binding_affinity_kcal_mol"] = [
            deltaG(pIC50) if not np.isnan(pIC50) else np.nan
            for pIC50 in mol_df["pIC50"]
        ]
        mol_df["exp_binding_affinity_kT"] = [
            deltaG_kT(pIC50) if not np.isnan(pIC50) else np.nan
            for pIC50 in mol_df["pIC50"]
        ]
        # Need to flip upper/lower bounds again
        mol_df["exp_binding_affinity_kcal_mol_95ci_lower"] = [
            deltaG(pIC50_upper) if not np.isnan(pIC50_upper) else np.nan
            for pIC50_upper in mol_df["pIC50_95ci_upper"]
        ]
        mol_df["exp_binding_affinity_kT_95ci_lower"] = [
            deltaG_kT(pIC50_upper) if not np.isnan(pIC50_upper) else np.nan
            for pIC50_upper in mol_df["pIC50_95ci_upper"]
        ]
        mol_df["exp_binding_affinity_kcal_mol_95ci_upper"] = [
            deltaG(pIC50_lower) if not np.isnan(pIC50_lower) else np.nan
            for pIC50_lower in mol_df["pIC50_95ci_lower"]
        ]
        mol_df["exp_binding_affinity_kT_95ci_upper"] = [
            deltaG_kT(pIC50_lower) if not np.isnan(pIC50_lower) else np.nan
            for pIC50_lower in mol_df["pIC50_95ci_lower"]
        ]
    # Based on already calculated dG values so can be the same for both
    mol_df["exp_binding_affinity_kcal_mol_stderr"] = [
        (
            abs(affinity_upper - affinity_lower) / 4.0
            if ((not np.isnan(affinity_lower)) and (not np.isnan(affinity_upper)))
            else np.nan
        )
        for _, (affinity_lower, affinity_upper) in mol_df[
            [
                "exp_binding_affinity_kcal_mol_95ci_lower",
                "exp_binding_affinity_kcal_mol_95ci_upper",
            ]
        ].iterrows()
    ]
    mol_df["exp_binding_affinity_kT_stderr"] = [
        (
            abs(affinity_upper - affinity_lower) / 4.0
            if ((not np.isnan(affinity_lower)) and (not np.isnan(affinity_upper)))
            else np.nan
        )
        for _, (affinity_lower, affinity_upper) in mol_df[
            [
                "exp_binding_affinity_kT_95ci_lower",
                "exp_binding_affinity_kT_95ci_upper",
            ]
        ].iterrows()
    ]

    # Keep only the best measurement for each molecule
    if keep_best_per_mol:

        def get_best_mol(g):
            if f"{assay_name}: Curve class" in g:
                sort_vals = [f"{assay_name}: Curve class", "pIC50_stderr"]
            else:
                sort_vals = ["pIC50_stderr"]
            g = g.sort_values(by=sort_vals, ascending=True)
            return g.iloc[0, :]

        mol_df = mol_df.groupby("name", as_index=False, group_keys=False).apply(
            get_best_mol
        )

    return mol_df


def get_sdf_fn_from_dataset(
    dataset: str,
    fragalysis_dir,
):
    fn = os.path.join(fragalysis_dir, f"{dataset}_0A/{dataset}_0A.sdf")
    if not os.path.exists(fn):
        print(f"File {fn} not found...")
        fn = None  # not sure what behaviour this should have
    return fn


def check_filelist_has_elements(
    filelist: Union[glob.glob, list], tag: Optional[str] = "untagged"
) -> None:
    """
    Check that a glob or list of files actually contains some elements - if not, raise
    an error.

    Parameters
    ----------
    filelist : Union[glob.glob, List]
        List of files or glob
    tag : Optional[str]
        Tag to add to error message if list is empty

    Returns
    -------
    None
    """
    if len(filelist) == 0:
        raise ValueError(
            f"list of files or glob with tag: {tag} does not contain any elements"
        )


def is_valid_smiles(smiles):
    # Create an OEMol object
    mol = oechem.OEMol()

    # Attempt to parse the SMILES string
    if not oechem.OEParseSmiles(mol, smiles):
        return False

    # Check if the parsed molecule is valid
    if not mol.IsValid():
        return False

    return True


def combine_files(paths: list[Union[Path, str]], output_file):
    with open(output_file, "w") as ofs:
        for file_to_copy in paths:
            with open(file_to_copy) as file_to_copy_fd:
                ofs.write(file_to_copy_fd.read())


def check_name_length_and_truncate(name: str, max_length: int = 70, logger=None) -> str:
    # check for name length and truncate if necessary
    if len(name) > max_length:
        truncated_name = name[:max_length]
        if logger:
            logger.warning(
                f"Name {name} is longer than {max_length} characters and has been truncated to {truncated_name}, consider using shorter filenames"
            )
        return truncated_name
    else:
        return name


def check_empty_dataframe(
    df: pandas.DataFrame,
    logger=None,
    fail: str = "raise",
    tag: str = "",
    message: str = "",
) -> bool:
    if df.empty:
        if logger:
            logger.warning(f"Dataframe with tag: {tag} is empty due to: {message}")
        if fail == "raise":
            raise ValueError(f"Dataframe with tag: {tag} is empty due to: {message}")
        elif fail == "return":
            return True
        else:
            raise ValueError(f"fail argument {fail} not recognised")
