from typing import Dict, List
import pandas as pd
from pydantic import BaseModel, Field
import pickle as pkl
import numpy as np
import os
from .datasets.utils import load_openeye_sdf, get_ligand_rmsd_openeye
from openeye import oechem

## From FAH #####################################################################
class Model(BaseModel):
    class Config:
        allow_mutation = False
        extra = "forbid"


class ExperimentalCompoundData(Model):

    compound_id: str = Field(
        None,
        description="The unique compound identifier (PostEra or enumerated ID)",
    )

    smiles: str = Field(
        None,
        description="OpenEye canonical isomeric SMILES string defining suspected SMILES of racemic mixture (with unspecified stereochemistry) or specific enantiopure compound (if racemic=False); may differ from what is registered under compound_id.",
    )

    racemic: bool = Field(
        False,
        description="If True, this experiment was performed on a racemate; if False, the compound was enantiopure.",
    )

    achiral: bool = Field(
        False,
        description="If True, this compound has no chiral centers or bonds, by definition enantiopure",
    )

    absolute_stereochemistry_enantiomerically_pure: bool = Field(
        False,
        description="If True, the compound was enantiopure and stereochemistry recorded in SMILES is correct",
    )

    relative_stereochemistry_enantiomerically_pure: bool = Field(
        False,
        description="If True, the compound was enantiopure, but unknown if stereochemistry recorded in SMILES is correct",
    )

    experimental_data: Dict[str, float] = Field(
        dict(),
        description='Experimental data fields, including "pIC50" and uncertainty (either "pIC50_stderr" or  "pIC50_{lower|upper}"',
    )


class ExperimentalCompoundDataUpdate(Model):
    """A bundle of experimental data for compounds (racemic or enantiopure)."""

    compounds: List[ExperimentalCompoundData]


################################################################################


class CrystalCompoundData(Model):

    smiles: str = Field(
        None,
        description="OpenEye canonical isomeric SMILES string defining suspected SMILES of racemic mixture (with unspecified stereochemistry) or specific enantiopure compound (if racemic=False); may differ from what is registered under compound_id.",
    )

    compound_id: str = Field(
        None, description="The unique compound identifier of the ligand."
    )

    dataset: str = Field(
        None,
        description='Dataset name from Fragalysis (name of structure).'
    )

    str_fn: str = Field(
        None,
        description='Filename of the PDB structure.'
    )

    sdf_fn: str = Field(
        None,
        description='Filename of the SDF file'
    )

class PDBStructure(Model):
    pdb_id: str = Field(
        None,
        description='PDB identification code.'
    )
    str_fn: str = Field(
        None,
        description='Filename of local PDB structure.'
    )

class EnantiomerPair(Model):
    active: ExperimentalCompoundData = Field(description="Active enantiomer.")
    inactive: ExperimentalCompoundData = Field(
        description="Inactive enantiomer."
    )


class EnantiomerPairList(Model):
    pairs: List[EnantiomerPair]


class DockingDataset():

    def __init__(self, pkl_fn, dir_path):
        self.pkl_fn = pkl_fn
        self.dir_path = dir_path

        for path in [self.pkl_fn, self.dir_path]:
            assert os.path.exists(path)

    def read_pkl(self):
        self.compound_ids, self.xtal_ids, self.res_ranks = pkl.load(open(self.pkl_fn, 'rb'))

    # def construct_df(self):
    #     pd.DataFrame({"Compound_ID": self.compound_ids,
    #                   "Crystal_ID": self.xtal_ids,
    #                   })

    def calculate_rmsds(self, fragalysis_directory):
        from .datasets.utils import load_openeye_sdf, get_ligand_rmsd_openeye
        assert os.path.exists(os.path.join(fragalysis_directory, ref_sdf_fn))
        pass

    def get_cmpd_dir_path(self, cmpd_id):
        ## make sure this directory exists
        cmpd_dir = os.path.join(self.dir_path, cmpd_id)
        print(cmpd_dir)
        assert os.path.exists(cmpd_dir)
        return cmpd_dir

    def organize_docking_results(self):
        ## Make lists we want to use to collect information about docking results
        cmpd_ids = []
        xtal_ids = []
        chain_ids = []
        mcss_ranks = []
        sdf_fns = []
        cmplx_ids = []

        ## dictionary of reference sdf files for each compound id
        ref_fn_dict = {}

        ## since each compound has its own directory, we can use that directory
        for cmpd_id in self.compound_ids:

            cmpd_dir = self.get_cmpd_path(cmpd_id)

            ## get list of files in this directory
            fn_list = os.listdir(cmpd_dir)

            ## Process this list into info
            ## TODO: use REGEX instead
            sdf_list = [fn for fn in fn_list if os.path.splitext(fn)[1] == '.sdf']

            ## For each sdf file in this list, get all the information
            ## This is very fragile to the file naming scheme
            for fn in sdf_list:
                info = fn.split(".")[0].split("_")
                xtal = info[3]
                chain = info[4]
                cmpd_id = info[7]

                ## the ligand re-docked to their original xtal have no mcss rank, so this will fail
                try:
                    mcss_rank = info[9]

                except:

                    ## however its a convenient way of identifying which is the original xtal
                    ref_xtal = xtal
                    ref_sdf_fn = f"{ref_xtal}_{chain}/{ref_xtal}_{chain}.sdf"

                    ## save the ref filename to the dictionary and make the mcss_rank -1
                    ref_fn_dict[cmpd_id] = ref_sdf_fn
                    mcss_rank = -1

                ## append all the stuff to lists!
                cmpd_ids.append(cmpd_id)
                xtal_ids.append(xtal)
                chain_ids.append(chain)
                mcss_ranks.append((mcss_rank))
                sdf_fns.append(fn)
                cmplx_ids.append(f"{cmpd_id}_{xtal}_{chain}_{mcss_rank}")

        ref_sdf_fns = [ref_fn_dict[cmpd_id] for cmpd_id in cmpd_ids]

        self.df = pd.DataFrame(
            {"Complex_ID": cmplx_ids,
                "Compound_ID": cmpd_ids,
             "Crystal_ID": xtal_ids,
             "Chain_ID": chain_ids,
             "MCSS_Rank": mcss_ranks,
             "SDF_Filename": sdf_fns,
             "Reference_SDF": ref_sdf_fns
             }
        )

    def calculate_rmsd_and_posit_score(self, fragalysis_dir):
        rmsds = []
        posit_scores = []
        docking_scores = []
        print(self.df.head().to_dict(orient='index'))
        for data in self.df.to_dict(orient='index').values():
            cmpd_id = data["Compound_ID"]
            sdf_fn = data["SDF_Filename"]
            ref_fn = data["Reference_SDF"]

            cmpd_dir = self.get_cmpd_path(cmpd_id)

            sdf_path = os.path.join(cmpd_dir, sdf_fn)
            ref_path = os.path.join(fragalysis_dir, ref_fn)

            print(f"Loading rmsd calc on {sdf_path} compared to {ref_path}")
            ref = load_openeye_sdf(ref_fn)
            mobile = load_openeye_sdf(sdf_fn)

            posit_scores.append(oechem.OEGetSDData(mobile, "POSIT::Probability"))
            docking_scores.append(oechem.OEGetSDData(mobile, "Chemgauss4"))

            rmsd = get_ligand_rmsd_openeye(ref, mobile)
            rmsds.append(rmsd)

        self.df["POSIT"] = posit_scores
        self.df["Chemgauss4"] = docking_scores
        self.df["RMSD"] = rmsds

    def write_csv(self, output_csv_fn):
        self.df.to_csv(output_csv_fn, index=False)

    def analyze_docking_results(self, fragalysis_dir, output_csv_fn, test=False):
        self.organize_docking_results()
        # self.to_df()
        if test:
            self.df = self.df.head()
        self.calculate_rmsd_and_posit_score(fragalysis_dir)
        self.write_csv(output_csv_fn=output_csv_fn)




