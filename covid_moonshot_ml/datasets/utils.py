import json
import os.path

import numpy as np
import pandas
import re
from openeye import oechem

from ..schema import ExperimentalCompoundData, ExperimentalCompoundDataUpdate, \
    PDBStructure, CrystalCompoundData, EnantiomerPairList

MPRO_SEQRES = """\
SEQRES   1 A  306  SER GLY PHE ARG LYS MET ALA PHE PRO SER GLY LYS VAL
SEQRES   2 A  306  GLU GLY CYS MET VAL GLN VAL THR CYS GLY THR THR THR
SEQRES   3 A  306  LEU ASN GLY LEU TRP LEU ASP ASP VAL VAL TYR CYS PRO
SEQRES   4 A  306  ARG HIS VAL ILE CYS THR SER GLU ASP MET LEU ASN PRO
SEQRES   5 A  306  ASN TYR GLU ASP LEU LEU ILE ARG LYS SER ASN HIS ASN
SEQRES   6 A  306  PHE LEU VAL GLN ALA GLY ASN VAL GLN LEU ARG VAL ILE
SEQRES   7 A  306  GLY HIS SER MET GLN ASN CYS VAL LEU LYS LEU LYS VAL
SEQRES   8 A  306  ASP THR ALA ASN PRO LYS THR PRO LYS TYR LYS PHE VAL
SEQRES   9 A  306  ARG ILE GLN PRO GLY GLN THR PHE SER VAL LEU ALA CYS
SEQRES  10 A  306  TYR ASN GLY SER PRO SER GLY VAL TYR GLN CYS ALA MET
SEQRES  11 A  306  ARG PRO ASN PHE THR ILE LYS GLY SER PHE LEU ASN GLY
SEQRES  12 A  306  SER CYS GLY SER VAL GLY PHE ASN ILE ASP TYR ASP CYS
SEQRES  13 A  306  VAL SER PHE CYS TYR MET HIS HIS MET GLU LEU PRO THR
SEQRES  14 A  306  GLY VAL HIS ALA GLY THR ASP LEU GLU GLY ASN PHE TYR
SEQRES  15 A  306  GLY PRO PHE VAL ASP ARG GLN THR ALA GLN ALA ALA GLY
SEQRES  16 A  306  THR ASP THR THR ILE THR VAL ASN VAL LEU ALA TRP LEU
SEQRES  17 A  306  TYR ALA ALA VAL ILE ASN GLY ASP ARG TRP PHE LEU ASN
SEQRES  18 A  306  ARG PHE THR THR THR LEU ASN ASP PHE ASN LEU VAL ALA
SEQRES  19 A  306  MET LYS TYR ASN TYR GLU PRO LEU THR GLN ASP HIS VAL
SEQRES  20 A  306  ASP ILE LEU GLY PRO LEU SER ALA GLN THR GLY ILE ALA
SEQRES  21 A  306  VAL LEU ASP MET CYS ALA SER LEU LYS GLU LEU LEU GLN
SEQRES  22 A  306  ASN GLY MET ASN GLY ARG THR ILE LEU GLY SER ALA LEU
SEQRES  23 A  306  LEU GLU ASP GLU PHE THR PRO PHE ASP VAL VAL ARG GLN
SEQRES  24 A  306  CYS SER GLY VAL THR PHE GLN
SEQRES   1 B  306  SER GLY PHE ARG LYS MET ALA PHE PRO SER GLY LYS VAL
SEQRES   2 B  306  GLU GLY CYS MET VAL GLN VAL THR CYS GLY THR THR THR
SEQRES   3 B  306  LEU ASN GLY LEU TRP LEU ASP ASP VAL VAL TYR CYS PRO
SEQRES   4 B  306  ARG HIS VAL ILE CYS THR SER GLU ASP MET LEU ASN PRO
SEQRES   5 B  306  ASN TYR GLU ASP LEU LEU ILE ARG LYS SER ASN HIS ASN
SEQRES   6 B  306  PHE LEU VAL GLN ALA GLY ASN VAL GLN LEU ARG VAL ILE
SEQRES   7 B  306  GLY HIS SER MET GLN ASN CYS VAL LEU LYS LEU LYS VAL
SEQRES   8 B  306  ASP THR ALA ASN PRO LYS THR PRO LYS TYR LYS PHE VAL
SEQRES   9 B  306  ARG ILE GLN PRO GLY GLN THR PHE SER VAL LEU ALA CYS
SEQRES  10 B  306  TYR ASN GLY SER PRO SER GLY VAL TYR GLN CYS ALA MET
SEQRES  11 B  306  ARG PRO ASN PHE THR ILE LYS GLY SER PHE LEU ASN GLY
SEQRES  12 B  306  SER CYS GLY SER VAL GLY PHE ASN ILE ASP TYR ASP CYS
SEQRES  13 B  306  VAL SER PHE CYS TYR MET HIS HIS MET GLU LEU PRO THR
SEQRES  14 B  306  GLY VAL HIS ALA GLY THR ASP LEU GLU GLY ASN PHE TYR
SEQRES  15 B  306  GLY PRO PHE VAL ASP ARG GLN THR ALA GLN ALA ALA GLY
SEQRES  16 B  306  THR ASP THR THR ILE THR VAL ASN VAL LEU ALA TRP LEU
SEQRES  17 B  306  TYR ALA ALA VAL ILE ASN GLY ASP ARG TRP PHE LEU ASN
SEQRES  18 B  306  ARG PHE THR THR THR LEU ASN ASP PHE ASN LEU VAL ALA
SEQRES  19 B  306  MET LYS TYR ASN TYR GLU PRO LEU THR GLN ASP HIS VAL
SEQRES  20 B  306  ASP ILE LEU GLY PRO LEU SER ALA GLN THR GLY ILE ALA
SEQRES  21 B  306  VAL LEU ASP MET CYS ALA SER LEU LYS GLU LEU LEU GLN
SEQRES  22 B  306  ASN GLY MET ASN GLY ARG THR ILE LEU GLY SER ALA LEU
SEQRES  23 B  306  LEU GLU ASP GLU PHE THR PRO PHE ASP VAL VAL ARG GLN
SEQRES  24 B  306  CYS SER GLY VAL THR PHE GLN
"""


def add_seqres(pdb_in, pdb_out=None):
    """
    Add SARS-CoV2 MPRO residue sequence to PDB header.

    Parameters
    ----------
    pdb_in : str
        Input PDB file.
    pdb_out : str, optional
        Output PDB file. If not given, appends _seqres to the input file.
    """

    pdbfile_lines = [line for line in open(pdb_in, "r") if "UNK" not in line]
    pdbfile_lines = [line for line in pdbfile_lines if "LINK" not in line]
    ## Fix bad CL atom names
    pdbfile_lines = [re.sub("CL", "Cl", l) for l in pdbfile_lines]
    # # remove ligand hetatoms
    # pdbfile_lines = [ line for line in pdbfile_lines if 'LIG' not in line ]
    pdbfile_contents = "".join(pdbfile_lines)
    if not "SEQRES" in pdbfile_contents:
        pdbfile_contents = MPRO_SEQRES + pdbfile_contents

    if pdb_out is None:
        pdb_out = f"{pdb_in[:-4]}_seqres.pdb"
    with open(pdb_out, "w") as fp:
        fp.write(pdbfile_contents)

    print(f"Wrote {pdb_out}", flush=True)


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

def get_sdf_fn_from_dataset(dataset:str,
                                 fragalysis_dir,
                                 ):
    fn = os.path.join(fragalysis_dir, f"{dataset}_0A/{dataset}_0A.sdf")
    if not os.path.exists(fn):
        print(f"File {fn} not found...")
        fn = None ## not sure what behaviour this should have
    return fn

def parse_experimental_compound_data(
        exp_fn: str,
        json_fn: str
):
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

def parse_fragalysis_data(frag_fn,
                          x_dir,
                          cmpd_ids=None,
                          o_dir=False):
    ## Load in csv
    sars2_structures = pandas.read_csv(frag_fn).fillna("")

    if cmpd_ids is not None:
        ## Filter fragalysis dataset by the compounds we want to test
        sars2_filtered = sars2_structures[sars2_structures['Compound ID'].isin(cmpd_ids)]
    else:
        sars2_filtered = sars2_structures

    if o_dir:
        mols_wo_sars2_xtal = sars2_filtered[sars2_filtered["Dataset"].isna()][["Compound ID", "SMILES", "Dataset"]]
        mols_w_sars2_xtal = sars2_filtered[~sars2_filtered["Dataset"].isna()][["Compound ID", "SMILES", "Dataset"]]

        ## Use utils function to get sdf file from dataset
        mols_w_sars2_xtal["SDF"] = mols_w_sars2_xtal["Dataset"].apply(get_sdf_fn_from_dataset,
                                                                      fragalysis_dir=x_dir)

        ## Save csv files for each dataset
        mols_wo_sars2_xtal.to_csv(os.path.join(o_dir, "mers_ligands_without_SARS2_structures.csv"),
                                  index=False)

        mols_w_sars2_xtal.to_csv(os.path.join(o_dir, "mers_ligands_with_SARS2_structures.csv"),
                                 index=False)

    ## Construct sars_xtal list
    sars_xtals = {}
    for data in sars2_filtered.to_dict('index').values():
        cmpd_id = data["Compound ID"]
        dataset = data["Dataset"]
        print(cmpd_id, dataset)
        if len(dataset) > 0:
            if not sars_xtals.get(cmpd_id) or '-P' in dataset:
                sars_xtals[cmpd_id] = CrystalCompoundData(
                    smiles=data["SMILES"],
                    compound_id=cmpd_id,
                    dataset=dataset,
                    sdf_fn=get_sdf_fn_from_dataset(dataset, x_dir)
                )
        else:
            sars_xtals[cmpd_id] = CrystalCompoundData()
    print(sars_xtals.items())

    for cmpd_id, xtal in sars_xtals.items():
        print(xtal.compound_id, xtal.dataset)

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

    return(compound_to_xtals, xtal_to_compound)


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

def split_openeye_mol(complex_mol: oechem.OEMolBase):
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
    oechem.OESplitMolComplex(lig_mol, prot_mol, water_mol, oth_mol, complex_mol)

    return {'complex': complex_mol,
            'lig': lig_mol,
            'pro': prot_mol,
            'water': water_mol,
            'other': oth_mol}

def get_ligand_rmsd_openeye(ref: oechem.OEMolBase,
                    mobile: oechem.OEMolBase):

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


    rmsd_array = md.rmsd(ref_lig,
                         mobile_lig,
                         precentered=True
                         )
    per_res_rmsd = rmsd_array[0] / ref_lig.n_atoms
    #
    rmsd_array2 = md.rmsd(ref,
                         mobile,
                         atom_indices=mobile_idx,
                          ref_atom_indices=ref_idx
                         )
    print(rmsd_array,
          per_res_rmsd,
          rmsd_array2)


