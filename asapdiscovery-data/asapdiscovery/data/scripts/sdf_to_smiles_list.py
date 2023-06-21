"""
Load in a multi-ligand SDF file and write out a txt file with the format `SMILES <space> ligand_name`
Ligand name expected to be in the Compound_ID field of the sdf file.
"""
from argparse import ArgumentParser
from pathlib import Path

from asapdiscovery.data.logging import FileLogger
from asapdiscovery.data.openeye import load_openeye_sdfs, oechem


def get_args():
    parser = ArgumentParser(description="")
    parser.add_argument(
        "-s",
        "--sdf_fn",
        required=True,
        type=str,
        help="SDF file containing molecules of interest.",
    )
    parser.add_argument(
        "-o", "--output_fn", required=True, type=Path, help="Output file."
    )
    return parser.parse_args()


def main():
    args = get_args()

    logger = FileLogger(
        logname="sdf_to_smiles_list", path=args.output_fn.parent
    ).getLogger()

    logger.info(f"Loading SDF file: {args.sdf_fn}")

    mols = load_openeye_sdfs(args.sdf_fn)

    logger.info(f"Loaded {len(mols)} molecules.")

    output_lines = []
    for i, mol in enumerate(mols):
        # Get the compound ID from the SD data if it exists, otherwise use the index
        compound_id = oechem.OEGetSDData(mol, "Compound_ID").rstrip()
        if not compound_id:
            compound_id = f"compound_{i}"
            logger.warning(
                f"No Compound_ID found for '{compound_id}', using index instead."
            )

        # Get the SMILES string from the SD data if it exists, otherwise generate it
        smiles = oechem.OEGetSDData(mol, "SMILES")
        if not smiles:
            logger.warning(
                f"No SMILES string found for {compound_id}, generating one from molecule instead."
            )
            smiles = oechem.OEMolToSmiles(mol)

        output_lines.append(f"{smiles} {compound_id}\n")

    logger.info(f"Writing SMILES strings to: {args.output_fn}")
    with open(args.output_fn, "w") as f:
        for line in list(set(output_lines)):
            f.write(line)


if __name__ == "__main__":
    main()
