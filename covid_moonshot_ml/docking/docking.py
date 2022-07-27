import os

from ..schema import CrystalCompoundData, ExperimentalCompoundData, \
    PDBStructure, ExperimentalCompoundDataUpdate
from ..datasets.utils import get_sdf_fn_from_dataset
from ..datasets.pdb import load_pdbs_from_yaml

def build_docking_systems(
    exp_compounds, xtal_compounds, compound_idxs, n_top=1
):
    """
    Build systems to run through docking.
    Parameters
    ----------
    exp_compounds : list[schema.ExperimentalCompoundData]
        List of compounds to dock
    xtal_compounds : list[schema.CrystalCompoundData]
        List of all crystal structures
    compound_idxs : list[int]
        List giving the index of the crystal structure to dock to for each
        ligand. Should be the same length as `exp_compounds`
    n_top : int, default=1
        Dock to top `n_top` crystal structures

    Returns
    -------
    list[kinoml.core.systems.ProteinLigandComplex]
        List of protein+ligand systems for docking
    """
    from kinoml.core.proteins import Protein
    from kinoml.core.ligands import Ligand
    from kinoml.core.systems import ProteinLigandComplex

    systems = []
    for (c, idx) in zip(exp_compounds, compound_idxs):
        ## Make sure that there are enough crystal structures to dock to
        n_dock = min(n_top, len(idx))
        for i in range(n_dock):
            ## Build protein, ligand, and complex objects
            x = xtal_compounds[idx[i]]
            protein = Protein.from_file(x.str_fn, name="MPRO")
            protein.chain_id = x.str_fn.split("_")[-2][-1]
            protein.expo_id = "LIG"
            ligand = Ligand.from_smiles(smiles=c.smiles, name=c.compound_id)
            systems.append(ProteinLigandComplex(components=[protein, ligand]))

    return systems


def build_combined_protein_system_from_sdf(pdb_fn, sdf_fn):
    protein = Protein.from_file(pdb_fn, name="MERS-Mpro")
    ligand = Ligand.from_file(sdf_fn)
    return ProteinLigandComplex


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
                          cmpd_ids,
                          o_dir=False):
    ## Load in csv
    sars2_structures = pandas.read_csv(frag_fn).fillna("")

    ## Filter fragalysis dataset by the compounds we want to test
    sars2_filtered = sars2_structures[sars2_structures['Compound ID'].isin(cmpd_ids)]

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




def parse_xtal(x_fn, x_dir):
    """
    Load all crystal structures into schema.CrystalCompoundData objects.
    Parameters
    ----------
    x_fn : str
        CSV file giving information on each crystal structure
    x_dir : str
        Path to directory containing directories with crystal structure PDB
        files
    Returns
    -------
    List[schema.CrystalCompoundData]
        List of parsed crystal structures
    """
    import pandas

    from ..schema import CrystalCompoundData

    df = pandas.read_csv(x_fn)

    ## Find all P-files
    idx = [(type(d) is str) and ("-P" in d) for d in df["Dataset"]]

    ## Build argument dicts for the CrystalCompoundData objects
    xtal_dicts = [
        dict(zip(("smiles", "dataset"), r[1].values))
        for r in df.loc[idx, ["SMILES", "Dataset"]].iterrows()
    ]

    ## Add structure filename information
    for d in xtal_dicts:
        fn_base = (
            f'{x_dir}/{d["dataset"]}_0{{}}/{d["dataset"]}_0{{}}_' "seqres.pdb"
        )
        fn = fn_base.format("A", "A")
        if os.path.isfile(fn):
            d["str_fn"] = fn
        else:
            fn = fn_base.format("B", "B")
            assert os.path.isfile(fn), f'No structure found for {d["dataset"]}.'
            d["str_fn"] = fn

    ## Build CrystalCompoundData objects for each row
    xtal_compounds = [CrystalCompoundData(**d) for d in xtal_dicts]

    return xtal_compounds


def run_docking(cache_dir, output_dir, loop_db, n_procs, docking_systems):
    from kinoml.features.complexes import OEDockingFeaturizer

    featurizer = OEDockingFeaturizer(
        cache_dir=cache_dir,
        output_dir=output_dir,
        loop_db=loop_db,
        n_processes=n_procs,
    )
    docking_systems = featurizer.featurize(docking_systems)

    return docking_systems
