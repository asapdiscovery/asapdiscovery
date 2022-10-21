import os.path
from openeye import oechem
import numpy as np
import pandas
import re
import rdkit.Chem as Chem

from ..schema import (
    ExperimentalCompoundData,
    ExperimentalCompoundDataUpdate,
    CrystalCompoundData,
    EnantiomerPair,
    EnantiomerPairList,
)


def add_seqres(pdb_in, seqres_str=None, dbref_str=None, pdb_out=None):
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
    # # remove ligand hetatoms
    # pdbfile_lines = [ line for line in pdbfile_lines if 'LIG' not in line ]
    if seqres_str:
        pdbfile_lines = [line for line in pdbfile_lines if not "SEQRES" in line]
        pdbfile_contents = "".join(pdbfile_lines)
        # seqres_str +
    else:
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


def load_openeye_pdb(pdb_fn, alt_loc=False):
    ifs = oechem.oemolistream()
    ifs_flavor = oechem.OEIFlavor_PDB_Default | oechem.OEIFlavor_PDB_DATA
    ## Add option for keeping track of alternat locations in PDB file
    if alt_loc:
        ifs_flavor |= oechem.OEIFlavor_PDB_ALTLOC
    ifs.SetFlavor(
        oechem.OEFormat_PDB,
        ifs_flavor,
    )
    ifs.open(pdb_fn)
    in_mol = oechem.OEGraphMol()
    oechem.OEReadMolecule(ifs, in_mol)
    ifs.close()

    return in_mol


def load_openeye_sdf(sdf_fn):
    ifs = oechem.oemolistream()
    ifs.SetFlavor(
        oechem.OEFormat_SDF,
        oechem.OEIFlavor_SDF_Default,
    )
    ifs.open(sdf_fn)
    coords_mol = oechem.OEGraphMol()
    oechem.OEReadMolecule(ifs, coords_mol)
    ifs.close()

    return coords_mol


def save_openeye_pdb(mol, pdb_fn):
    ofs = oechem.oemolostream()
    ofs.SetFlavor(oechem.OEFormat_PDB, oechem.OEOFlavor_PDB_Default)
    ofs.open(pdb_fn)
    oechem.OEWriteMolecule(ofs, mol)
    ofs.close()


def save_openeye_sdf(mol, sdf_fn):
    ofs = oechem.oemolostream()
    ofs.SetFlavor(oechem.OEFormat_SDF, oechem.OEOFlavor_SDF_Default)
    ofs.open(sdf_fn)
    oechem.OEWriteMolecule(ofs, mol)
    ofs.close()


def split_openeye_mol(complex_mol, lig_chain="A", prot_cutoff_len=10):
    """
    Split an OpenEye-loaded molecule into protein, ligand, etc.

    Parameters
    ----------
    complex_mol : oechem.OEMolBase
        Complex molecule to split.
    lig_chain : str, default="A"
        Which copy of the ligand to keep. Pass None to keep all ligand atoms.
    prot_cutoff_len : int, default=10
        Minimum number of residues in a protein chain required in order to keep

    Returns
    -------
    """

    ## Test splitting
    lig_mol = oechem.OEGraphMol()
    prot_mol = oechem.OEGraphMol()
    water_mol = oechem.OEGraphMol()
    oth_mol = oechem.OEGraphMol()

    ## Make splitting split out covalent ligands
    ## TODO: look into different covalent-related options here
    opts = oechem.OESplitMolComplexOptions()
    opts.SetSplitCovalent(True)
    opts.SetSplitCovalentCofactors(True)

    ## Select protein as all protein atoms in chain A or chain B
    prot_only = oechem.OEMolComplexFilterFactory(
        oechem.OEMolComplexFilterCategory_Protein
    )
    a_chain = oechem.OERoleMolComplexFilterFactory(
        oechem.OEMolComplexChainRoleFactory("A")
    )
    b_chain = oechem.OERoleMolComplexFilterFactory(
        oechem.OEMolComplexChainRoleFactory("B")
    )
    a_or_b_chain = oechem.OEOrRoleSet(a_chain, b_chain)
    opts.SetProteinFilter(oechem.OEAndRoleSet(prot_only, a_or_b_chain))

    ## Select ligand as all residues with resn LIG
    lig_only = oechem.OEMolComplexFilterFactory(
        oechem.OEMolComplexFilterCategory_Ligand
    )
    if lig_chain is None:
        opts.SetLigandFilter(lig_only)
    else:
        lig_chain = oechem.OERoleMolComplexFilterFactory(
            oechem.OEMolComplexChainRoleFactory(lig_chain)
        )
        opts.SetLigandFilter(oechem.OEAndRoleSet(lig_only, lig_chain))

    ## Set water filter (keep all waters in A, B, or W chains)
    ##  (is this sufficient? are there other common water chain ids?)
    wat_only = oechem.OEMolComplexFilterFactory(
        oechem.OEMolComplexFilterCategory_Water
    )
    w_chain = oechem.OERoleMolComplexFilterFactory(
        oechem.OEMolComplexChainRoleFactory("W")
    )
    all_wat_chains = oechem.OEOrRoleSet(a_or_b_chain, w_chain)
    opts.SetWaterFilter(oechem.OEAndRoleSet(wat_only, all_wat_chains))

    oechem.OESplitMolComplex(
        lig_mol,
        prot_mol,
        water_mol,
        oth_mol,
        complex_mol,
        opts,
    )

    prot_mol = trim_small_chains(prot_mol, prot_cutoff_len)

    return {
        "complex": complex_mol,
        "lig": lig_mol,
        "pro": prot_mol,
        "water": water_mol,
        "other": oth_mol,
    }


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
