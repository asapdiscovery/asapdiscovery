"""
Load in a multi-ligand SDF file and write out a txt file with the format `SMILES <space> ligand_name`
Ligand name expected to be in the Compound_ID field of the sdf file. 
"""
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description="")
    parser.add_argument(
        "-s",
        "--sdf_fn",
        required=True,
        help="SDF file containing molecules of interest.",
    )
    parser.add_argument("-o", "--output_fn", required=True, help="Output file.")
    return parser.parse_args()


def main():
    args = get_args()
    from asapdiscovery.data.openeye import load_openeye_sdfs, oechem

    mols = load_openeye_sdfs(args.sdf_fn)
    output_lines = [
        f"{oechem.OEGetSDData(mol, 'SMILES')} {oechem.OEGetSDData(mol, 'Compound_ID').rstrip()}\n"
        for mol in mols
    ]

    with open(args.output_fn, "w") as f:
        for line in list(set(output_lines)):
            f.write(line)


if __name__ == "__main__":
    main()
