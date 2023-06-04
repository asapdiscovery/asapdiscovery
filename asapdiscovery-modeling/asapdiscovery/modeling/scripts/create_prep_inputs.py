import argparse
from pathlib import Path

from asapdiscovery.data.schema import CrystalCompoundData, CrystalCompoundDataset
from asapdiscovery.data.utils import check_filelist_has_elements
from asapdiscovery.modeling.schema import MoleculeFilter, PreppedTarget, PreppedTargets

# TODO: Add a function to check if the input file is a CrystalCompoundDataset
# TODO: Probably want to move to requiring that the input is a CrystalCompoundDataset


def get_args():
    parser = argparse.ArgumentParser(description="")

    # Input arguments
    parser.add_argument(
        "-d",
        "--structure_dir",
        type=Path,
        required=False,
        help="Path to downloaded input files",
    )
    parser.add_argument(
        "-i",
        "--input_file",
        type=Path,
        required=False,
        help="Path to serialized CrystalCompoundDataset.",
    )
    parser.add_argument(
        "-o", "--output_dir", type=Path, required=True, help="Path to output_dir."
    )
    parser.add_argument(
        "--components_to_keep", type=str, nargs="+", default=["protein", "ligand"]
    )
    parser.add_argument("--active_site_chain", type=str, default="A")
    parser.add_argument("--ligand_chain", type=str, default="A")
    parser.add_argument("--protein_chains", type=str, default=[], help="")
    parser.add_argument(
        "--oe_active_site_residue",
        type=str,
        default=None,
        help="OpenEye formatted site residue for active site identification, i.e. 'HIS:41: :A:0: '",
    )
    return parser.parse_args()


def main():
    args = get_args()

    if "ligand" not in args.components_to_keep and not args.oe_active_site_residue:
        raise ValueError(
            f"components_to_keep: {args.components_to_keep} do not include 'ligand' and no oe_active_site_residue provided.\n"
            "Must provide OpenEye formatted oe_active_site_residue if not keeping ligand."
        )

    args.output_dir.mkdir(exist_ok=True, parents=True)
    if args.input_file:
        if args.input_file.suffix == ".csv":
            xtals = CrystalCompoundDataset.from_csv(args.input_file).iterable
        elif args.input_file.suffix == ".pkl":
            xtals = CrystalCompoundDataset.from_pkl(args.input_file).iterable
        else:
            raise NotImplementedError

    elif args.structure_dir:
        protein_files = list(args.structure_dir.glob("*"))
        check_filelist_has_elements(protein_files)
        xtals = [
            CrystalCompoundData(dataset=protein_file.stem, str_fn=str(protein_file))
            for protein_file in protein_files
            if protein_file.suffix == ".pdb" or protein_file.suffix == ".cif"
        ]
    else:
        raise ValueError("Must provide either structure_dir or input_file.")

    targets = []
    for xtal in xtals:
        # TODO: Make this a bit more clever
        output_tags = []
        if xtal.dataset:
            output_tags.append(xtal.dataset)
        if xtal.compound_id:
            output_tags.append(xtal.compound_id)
        output_name = "_".join(output_tags)

        targets.append(
            PreppedTarget(
                source=xtal,
                output_name=output_name,
                active_site_chain=args.active_site_chain,
                oe_active_site_residue=args.oe_active_site_residue,
                molecule_filter=MoleculeFilter(
                    components_to_keep=args.components_to_keep,
                    ligand_chain=args.ligand_chain,
                    protein_chains=args.protein_chains,
                ),
            )
        )
    PreppedTargets.from_list(targets).to_pkl(args.output_dir / "to_prep.pkl")
