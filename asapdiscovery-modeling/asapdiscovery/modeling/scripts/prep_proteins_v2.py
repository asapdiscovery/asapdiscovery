"""
Prepare proteins.
Creates oedu binary DesignUnit files, complex and protein-only pdb files, and ligand-only sdf files.

Example Usage:
"""

from argparse import ArgumentParser
from pathlib import Path
from asapdiscovery.data.schema import CrystalCompoundData, CrystalCompoundDataset


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

    # Model Building Options
    parser.add_argument(
        "-l",
        "--loop_db",
        default=Path(
            "/Users/alexpayne/Scientific_Projects/mers-drug-discovery/spruce_bace.loop_db"
        ),
        help="Path to loop database.",
    )
    parser.add_argument(
        "-s",
        "--seqres_yaml",
        default=Path("../../../../metadata/mpro_mers_seqres.yaml"),
        help="Path to yaml file of SEQRES.",
    )
    parser.add_argument(
        "--protein_only",
        action="store_true",
        default=True,
        help="If true, generate design units with only the protein in them",
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
    # Load structure
    if xtal.str_fn.suffix == ".cif1":
        from asapdiscovery.data.openeye import load_openeye_cif1

        prot = load_openeye_cif1(xtal.str_fn)
    elif xtal.str_fn.suffix == ".pdb":
        from asapdiscovery.data.openeye import load_openeye_pdb

        prot = load_openeye_pdb(xtal.str_fn)

    # Align to reference
    if args.ref_prot:
        from asapdiscovery.modeling.modeling import align_receptor

        ref_path = Path(args.ref_prot)
        prot = align_receptor(
            initial_complex=prot,
            ref_prot=str(ref_path),
            dimer=True,
            split_initial_complex=True,
            mobile_chain="A",  # TODO: make this not hardcoded? not sure what logic to use though
            ref_chain="A",
        )

    # Build model
    xtal.build_model(
        loop_db=args.loop_db,
        seqres_yaml=args.seqres_yaml,
        protein_only=args.protein_only,
    )

    # Save
    xtal.save_design_unit(args.output_dir / f"{xtal.rcsb_id}.oedu")
    xtal.save_pdb(args.output_dir / f"{xtal.rcsb_id}.pdb")
    xtal.save_sdf(args.output_dir / f"{xtal.rcsb_id}.sdf")
    return xtal


def main():
    # Load structures
    args = parse_args()
    dataset = CrystalCompoundDataset()
    dataset.from_csv(args.prep_csv)
    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True)
    dataset.to_csv(args.output_dir / "prepped_structures.csv")

    #


if __name__ == "__main__":
    main()
