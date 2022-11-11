from openeye import oechem
import pandas
import re
from asap_datasets.schema import (
    ExperimentalCompoundData,
    ExperimentalCompoundDataUpdate,
)

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


if __name__ == "__main__":
    filter_docking_inputs()
