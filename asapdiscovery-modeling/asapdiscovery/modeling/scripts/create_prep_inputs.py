import argparse
from pathlib import Path
from asapdiscovery.data.schema import CrystalCompoundData
from asapdiscovery.modeling.schema import PreppedTarget, PreppedTargets, MoleculeFilter


def get_args():
    parser = argparse.ArgumentParser(description="")

    # Input arguments
    parser.add_argument(
        "-d",
        "--structure_dir",
        type=Path,
        required=True,
        help="Path to downloaded input files",
    )
    parser.add_argument(
        "-o", "--output_dir", type=Path, required=True, help="Path to output_dir."
    )
    parser.add_argument(
        "--components_to_keep", type=list[str], nargs="+", default=["protein", "ligand"]
    )
    parser.add_argument("--active_site_chain", type=str, default="A")
    parser.add_argument("--ligand_chain", type=str, default=["A"])
    parser.add_argument("--protein_chains", type=str, default=["A", "B"])
    return parser.parse_args()


def main():
    args = get_args()
    protein_files = args.structure_dir.glob("*")
    targets = []
    for protein_file in protein_files:
        targets.append(
            PreppedTarget(
                source=CrystalCompoundData(str_fn=protein_file),
                output_name=protein_file.stem,
                active_site_chain=args.active_site_chain,
                molecule_filter=MoleculeFilter(
                    components_to_keep=args.components_to_keep,
                    ligand_chain=args.ligand_chain,
                    protein_chains=args.protein_chains,
                ),
            )
        )
    PreppedTargets.from_list(targets).to_pkl(args.output_dir / "to_prep.pkl")
