"""
Prepare proteins.
Combines ideas from the previous prep_proteins and prep_mers_pdbs scripts.
Requires a csv file generated using the CrystalCompoundDataset class; i.e. the columns of the csv file
must be the same as the attributes of the CrystalCompoundData class.
Creates oedu binary DesignUnit files, complex and protein-only pdb files, and ligand-only sdf files.

Example Usage:
"""

from argparse import ArgumentParser
from pathlib import Path

from asapdiscovery.data.schema import CrystalCompoundData, CrystalCompoundDataset
from asapdiscovery.modeling.modeling import spruce_protein


def parse_args():
    parser = ArgumentParser()

    # Input
    parser.add_argument(
        "-csv",
        "--prep_csv",
        type=Path,
        required=True,
        help="CSV file giving information of which structures to prep.",
    )

    # Alignment
    parser.add_argument(
        "-r",
        "--ref_prot",
        default=Path("../tests/prep_mers_rcsb/inputs/reference.pdb"),
        type=Path,
        help="Path to reference pdb to align to. If None, no alignment will be performed",
    )
    parser.add_argument(
        "--ref_chain", default="A", help="Chain of reference protein to align to."
    )
    parser.add_argument(
        "--mobile_chain",
        default="A",
        help="Chain of prepped protein to align to reference.",
    )

    # Model Building Options
    parser.add_argument(
        "-l",
        "--loop_db",
        default=None,
        type=Path,
        help="Path to loop database.",
    )
    parser.add_argument(
        "-s",
        "--seqres_yaml",
        default=None,
        type=Path,
        help="Path to yaml file of SEQRES.",
    )
    parser.add_argument(
        "--protein_only",
        action="store_true",
        default=False,
        help="If true, generate design units with only the protein in them",
    )
    parser.add_argument(
        "--keep_water", default=True, help="If true, keep crystallographic water."
    )

    # Output
    parser.add_argument("-o", "--output_dir", type=Path, required=True)

    # Performance Arguments
    parser.add_argument(
        "-n",
        "--num_cores",
        type=int,
        default=1,
        help="Number of concurrent processes to run.",
    )
    return parser.parse_args()


def prep_protein(xtal: CrystalCompoundData, args):
    """
    Prep a protein.
    """
    import logging
    from asapdiscovery.data.openeye import oechem

    # Set up logger
    name = xtal.output_name
    output_dir = args.output_dir / xtal.output_name

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    logfile = output_dir / f"{name}-log.txt"
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(str(logfile), mode="w")
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Set up logging for OE Functions to redirect Info / Warnings / etc from stdout
    errfs = oechem.oeofstream(str(output_dir / f"{name}-oelog.txt"))
    oechem.OEThrow.SetOutputStream(errfs)

    du_fn = output_dir / f"{name}-prepped_receptor_0.oedu"
    if du_fn.exists():
        logger.info(f"Already made {du_fn}!")
        return
    else:
        logger.info(f"Preparing {xtal.str_fn}")
    # Load structure
    fn = Path(xtal.str_fn)
    if fn.suffix == ".cif1":
        from asapdiscovery.data.openeye import load_openeye_cif1

        prot = load_openeye_cif1(xtal.str_fn)
    elif fn.suffix == ".pdb":
        from asapdiscovery.data.openeye import load_openeye_pdb

        prot = load_openeye_pdb(xtal.str_fn)
    else:
        raise ValueError(f"Unrecognized file type: {fn.suffix}")

    if xtal.lig_chain:
        from asapdiscovery.modeling.modeling import remove_extra_ligands

        logger.info(f"Removing ligands except for chain {xtal.lig_chain}")
        prot = remove_extra_ligands(prot, lig_chain=xtal.lig_chain)

    # Align to reference
    if args.ref_prot:
        from asapdiscovery.modeling.modeling import align_receptor

        prot = align_receptor(
            prot,
            dimer=True,
            ref_prot=str(args.ref_prot),
            split_initial_complex=args.protein_only,
            split_ref=True,
            ref_chain=args.ref_chain,
            mobile_chain=args.mobile_chain,
            keep_water=args.keep_water,
        )

    if args.seqres_yaml:
        import yaml
        from asapdiscovery.data.utils import seqres_to_res_list
        from asapdiscovery.modeling.modeling import mutate_residues

        with open(args.seqres_yaml) as f:
            seqres_dict = yaml.safe_load(f)
        seqres = seqres_dict["SEQRES"]

        res_list = seqres_to_res_list(seqres)
        logger.info("Making mutations")

        prot = mutate_residues(prot, res_list, place_h=True)
        seqres = " ".join(res_list)
    else:
        seqres = None

    logger.info("Sprucing protein")

    du = spruce_protein(
        initial_prot=prot,
        seqres=seqres,
        loop_db=args.loop_db,
        return_du=True,
        site_residue=xtal.active_site,
    )

    from asapdiscovery.data.openeye import oechem, save_openeye_pdb

    if type(du) == oechem.OEDesignUnit:
        logger.info("Saving Design Unit")

        du_fn = output_dir / f"{xtal.output_name}-prepped_receptor_0.oedu"
        oechem.OEWriteDesignUnit(str(du_fn), du)

        logger.info("Saving PDB")

        from asapdiscovery.data.openeye import (
            save_openeye_sdf,
            split_openeye_design_unit,
        )
        from asapdiscovery.modeling.modeling import add_seqres_to_openeye_protein

        # TODO: Make sure this doesn't fail if there is no ligand
        lig, prot, complex_ = split_openeye_design_unit(du, lig_title=xtal.compound_id)
        prot = add_seqres_to_openeye_protein(prot, seqres)
        complex_ = add_seqres_to_openeye_protein(complex_, seqres)

        prot_fn = output_dir / f"{xtal.output_name}-prepped_protein.pdb"
        save_openeye_pdb(prot, str(prot_fn))

        complex_fn = output_dir / f"{xtal.output_name}-prepped_complex.pdb"
        save_openeye_pdb(complex_, str(complex_fn))

        lig_fn = output_dir / f"{xtal.output_name}-prepped_ligand.sdf"
        save_openeye_sdf(lig, str(lig_fn))

    elif type(du) == oechem.OEGraphMol:
        logger.error("Design Unit preparation failed. Saving spruced protein")
        prot_fn = output_dir / f"{xtal.output_name}-failed-spruced.pdb"
        save_openeye_pdb(du, str(prot_fn))
    return xtal


def main():
    # Load structures
    args = parse_args()
    dataset = CrystalCompoundDataset()
    dataset.from_csv(args.prep_csv)
    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True)

    for xtal in dataset.structures:
        prep_protein(xtal, args)


if __name__ == "__main__":
    main()
