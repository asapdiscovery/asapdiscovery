"""
Create oedu binary DesignUnit files for input protein structures.
This was designed to be able to be used for either molecules from the PDB or from fragalysis.

Example Usage:
    python prep_proteins.py \
    -d /Users/alexpayne/lilac-mount-point/asap-datasets/mers_pdb_download \
    -p ../../../metadata/mers-structures-dimers.yaml \
    -r /Users/alexpayne/lilac-mount-point/asap-datasets/mpro_fragalysis_2022_10_12/extra_files/reference.pdb \
    -l /Users/alexpayne/lilac-mount-point/asap-datasets/rcsb_spruce.loop_db \
    -o /Users/alexpayne/lilac-mount-point/asap-datasets/mers_fauxalysis/mers_prepped_structures_dimers_only \
    -s ../../../metadata/mpro_mers_seqres.yaml \
    --protein_only
"""
import argparse
import multiprocessing as mp
import os
import sys
import yaml
import logging

from asapdiscovery.docking import prep_mp

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    )
)
from asapdiscovery.data.schema import CrystalCompoundData
from asapdiscovery.data import pdb
from asapdiscovery.data.fragalysis import parse_fragalysis


################################################################################
def get_args():
    parser = argparse.ArgumentParser(description="")

    ## Input arguments
    parser.add_argument(
        "-d",
        "--structure_dir",
        required=True,
        help="Path to fragalysis/aligned/ directory or directory to put PDB structures.",
    )

    parser.add_argument(
        "-x",
        "--xtal_csv",
        default=None,
        help="CSV file giving information of which structures to prep.",
    )
    parser.add_argument(
        "-p",
        "--pdb_yaml_path",
        default=None,
        help="Yaml file containing PDB IDs.",
    )

    parser.add_argument(
        "-r",
        "--ref_prot",
        default=None,
        type=str,
        help="Path to reference pdb to align to. If None, no alignment will be performed",
    )

    ## Output arguments
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="Path to output_dir.",
    )

    ## Model-building arguments
    parser.add_argument(
        "-l",
        "--loop_db",
        help="Path to loop database.",
    )
    parser.add_argument(
        "-s",
        "--seqres_yaml",
        help="Path to yaml file of SEQRES.",
    )
    parser.add_argument(
        "--protein_only",
        action="store_true",
        default=False,
        help="If true, generate design units with only the protein in them",
    )
    parser.add_argument(
        "--include_non_Pseries",
        default=False,
        action="store_true",
        help="If true, the p_only flag of parse_xtal will be set to False. Default is False, which sets p_only to True",
    )
    parser.add_argument(
        "--log_file",
        default="prep_proteins_log.txt",
        help="Path to high level log file.",
    )

    ## Performance arguments
    parser.add_argument(
        "-n",
        "--num_cores",
        type=int,
        default=1,
        help="Number of concurrent processes to run.",
    )

    return parser.parse_args()


def main():
    args = get_args()
    handler = logging.FileHandler(args.log_file, mode="w")
    main_logger = logging.getLogger("main")
    main_logger.setLevel(logging.INFO)
    main_logger.addHandler(handler)

    if args.xtal_csv:
        p_only = False if args.include_non_Pseries else True
        if p_only:
            xtal_compounds = parse_fragalysis(
                args.xtal_csv,
                args.structure_dir,
                name_filter="Mpro-P",
                drop_duplicate_datasets=True,
            )
        else:
            xtal_compounds = parse_fragalysis(
                args.xtal_csv,
                args.structure_dir,
            )

        for xtal in xtal_compounds:
            ## Get chain
            ## The parentheses in this string are the capture group

            xtal.output_name = f"{xtal.dataset}_{xtal.compound_id}"

            frag_chain = xtal.dataset[-2:]

            ## We also want the chain in the form of a single letter ('A', 'B'), etc
            xtal.active_site_chain = frag_chain[-1]

            ## If we aren't keeping the ligands, then we want to give it a site residue to use
            if args.protein_only:
                xtal.active_site = f"His:41: :{xtal.active_site_chain}"

    elif args.pdb_yaml_path:
        ## First, get list of pdbs from yaml file
        pdb_dict = pdb.load_pdbs_from_yaml(args.pdb_yaml_path)
        pdb_list = list(pdb_dict.keys())
        pdb.download_PDBs(pdb_list, args.structure_dir)
        active_site_chain = "A"
        ## If the yaml file doesn't have any options for the pdb file, then assume it is a dimer
        xtal_compounds = [
            CrystalCompoundData(
                str_fn=os.path.join(args.structure_dir, f"rcsb_{pdb_id}.pdb"),
                output_name=f"{pdb_id}_0{active_site_chain}",
                active_site_chain=active_site_chain,
                active_site=f"HIS:41: :{active_site_chain}",
                chains=values.get("chains", ["A", "B"]),
                oligomeric_state=values.get("oligomeric_state", "dimer"),
                protein_chains=values.get("protein_chains", ["A", "B"]),
            )
            for pdb_id, values in pdb_dict.items()
            if os.path.exists(
                os.path.join(args.structure_dir, f"rcsb_{pdb_id}.pdb")
            )
        ]
    else:
        raise NotImplementedError("Crystal CSV or PDB yaml file needed")

    if args.seqres_yaml:
        with open(args.seqres_yaml) as f:
            seqres_dict = yaml.safe_load(f)
        seqres = seqres_dict["SEQRES"]
    else:
        seqres = None

    mp_args = [
        (
            x,
            args.ref_prot,
            seqres,
            args.output_dir,
            args.loop_db,
            args.protein_only,
        )
        for x in xtal_compounds
    ]
    main_logger.info(mp_args[0])
    nprocs = min(mp.cpu_count(), len(mp_args), args.num_cores)
    main_logger.info(
        f"CPUS: {mp.cpu_count()}, Structure: {mp_args}, N Cores: {args.num_cores}"
    )
    main_logger.info(f"Prepping {len(mp_args)} structures over {nprocs} cores.")
    with mp.Pool(processes=nprocs) as pool:
        pool.starmap(prep_mp, mp_args)


if __name__ == "__main__":
    main()
