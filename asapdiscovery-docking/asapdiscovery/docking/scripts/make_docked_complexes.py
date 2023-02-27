"""
Make complex PDB files for docked SDF files.
"""
import argparse
import os
import sys

import pandas
from openeye import oechem

from asapdiscovery.data.openeye import load_openeye_pdb  # noqa: E402
from asapdiscovery.data.openeye import load_openeye_sdf  # noqa: 402
from asapdiscovery.data.openeye import save_openeye_pdb  # noqa: 402
from asapdiscovery.data.openeye import split_openeye_mol  # noqa: E402


########################################
def get_args():
    parser = argparse.ArgumentParser(description="")

    # I/O args
    parser.add_argument("-i", "--in_file", required=True, help="Input CSV file.")
    parser.add_argument("-d", "--xtal_dir", required=True, help="Fragalysis directory.")
    parser.add_argument(
        "-o",
        "--out_dir",
        help=(
            "Output directory. If not supplied, files will be generated in "
            "the same directory as the docked SDF file."
        ),
    )
    parser.add_argument(
        "-n", "--lig_name", default="LIG", help="Residue name for ligand."
    )

    # Selection args
    parser.add_argument(
        "-p",
        "--prot_only",
        action="store_true",
        help="Only keep protein atoms from the apo PDB file.",
    )

    return parser.parse_args()


def main():
    args = get_args()

    # Load docking results
    df = pandas.read_csv(args.in_file, index_col=0)

    # Dict to hold loaded Mpro structures to avoid reloading the same one a
    #  bunch of times
    xtal_structs = {}
    use_cols = ["ligand_id", "du_structure", "docked_file"]
    for _, (compound_id, struct, docked_fn) in df[use_cols].iterrows():
        try:
            xtal = xtal_structs[struct][0].CreateCopy()
            new_resid, new_atomid = xtal_structs[struct][1:]
        except KeyError:
            xtal_fn = f"{args.xtal_dir}/{struct}/{struct}_apo.pdb"
            xtal = load_openeye_pdb(xtal_fn)

            # Get rid of non-protein atoms
            if args.prot_only:
                xtal = split_openeye_mol(xtal, lig_chain=None)["pro"]

            ## Find max resid for numbering the ligand residue
            new_resid = (
                max([r.GetResidueNumber() for r in oechem.OEGetResidues(xtal)]) + 1
            )

            ## Same with atom numbering
            new_atomid = (
                max(
                    [
                        oechem.OEAtomGetResidue(a).GetSerialNumber()
                        for a in xtal.GetAtoms()
                    ]
                )
                + 1
            )

            ## Store data to avoid reloading/recalculating
            xtal_structs[struct] = (xtal.CreateCopy(), new_resid, new_atomid)

        try:
            mol = load_openeye_sdf(docked_fn)
        except TypeError:
            print(
                "Couldn't read SDF file:",
                compound_id,
                struct,
                docked_fn,
                flush=True,
            )
            continue

        num_elem_atoms = {}
        ## Adjust molecule residue properties
        for a in mol.GetAtoms():
            ## Set atom name
            cur_name = oechem.OEGetAtomicSymbol(a.GetAtomicNum())
            try:
                new_name = f"{cur_name}{num_elem_atoms[cur_name]}"
                num_elem_atoms[cur_name] += 1
            except KeyError:
                new_name = cur_name
                num_elem_atoms[cur_name] = 1
            a.SetName(new_name)

            # new_a = amap[a.GetIdx()]
            res = oechem.OEAtomGetResidue(a)
            res.SetName(args.lig_name.upper())
            res.SetResidueNumber(new_resid)
            res.SetSerialNumber(new_atomid)
            new_atomid += 1
            res.SetHetAtom(True)
            oechem.OEAtomSetResidue(a, res)

        ## First combine the mols, keeping track of mapping
        amap, bmap = oechem.OEAddMols(xtal, mol)

        out_base = f"{compound_id}_{struct}"
        if args.out_dir:
            out_fn = f"{args.out_dir}/{out_base}_bound.pdb"
        else:
            out_fn = f"{os.path.dirname(docked_fn)}/{out_base}_bound.pdb"
        save_openeye_pdb(xtal, out_fn)

        print(f"Wrote {out_fn}", flush=True)


if __name__ == "__main__":
    main()
