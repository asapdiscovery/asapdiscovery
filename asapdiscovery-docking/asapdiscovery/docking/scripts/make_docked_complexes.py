"""
Make complex PDB files for docked SDF files.
"""
import argparse
import os

import pandas
from asapdiscovery.data.openeye import combine_protein_ligand
from asapdiscovery.data.openeye import load_openeye_pdb  # noqa: E402
from asapdiscovery.data.openeye import load_openeye_sdf  # noqa: 402
from asapdiscovery.data.openeye import oechem
from asapdiscovery.data.openeye import save_openeye_pdb  # noqa: 402
from asapdiscovery.data.openeye import split_openeye_mol  # noqa: E402


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

    # Cache protein structures
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Cache protein structures to avoid loading the same structure many times.",
    )

    return parser.parse_args()


def main():
    args = get_args()

    # Load docking results
    df = pandas.read_csv(args.in_file, index_col=0)

    # Dict to hold loaded Mpro structure info to avoid recalculating
    xtal_structs = {}
    use_cols = ["ligand_id", "du_structure", "docked_file"]
    for _, (compound_id, struct, docked_fn) in df[use_cols].iterrows():
        print("Working on", compound_id, struct, docked_fn, flush=True)
        # Either haven't seen the structure yet (so need to load/calculate anyway),
        #  or aren't caching protein structures, so need to reload structure
        if (struct not in xtal_structs) or (not args.cache):
            # Load protein structure
            xtal_fn = f"{args.xtal_dir}/{struct}/{struct}_apo.pdb"
            xtal = load_openeye_pdb(xtal_fn)
            # Get rid of non-protein atoms
            if args.prot_only:
                xtal = split_openeye_mol(xtal, lig_chain=None)["pro"]

            # If we have seen the struct before (and therefore aren't caching)
            #  structures), we can just pull the calculated resid/atomid
            if struct in xtal_structs:
                new_resid, new_atomid = xtal_structs[struct]
            # Otherwise need to calculate resid/atomid, and store relevant info
            else:
                # Find max resid for numbering the ligand residue
                new_resid = (
                    max([r.GetResidueNumber() for r in oechem.OEGetResidues(xtal)]) + 1
                )

                # Same with atom numbering
                new_atomid = (
                    max(
                        [
                            oechem.OEAtomGetResidue(a).GetSerialNumber()
                            for a in xtal.GetAtoms()
                        ]
                    )
                    + 1
                )

                # Store protein structure if we're doing that, otherwise just resid
                #  and atomid
                if args.cache:
                    xtal_structs[struct] = (xtal.CreateCopy(), new_resid, new_atomid)
                else:
                    xtal_structs[struct] = (new_resid, new_atomid)
        else:
            # Already seen the protein and are caching structure
            xtal = xtal_structs[struct][0].CreateCopy()
            new_resid, new_atomid = xtal_structs[struct][1:]

        # Load ligand
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

        # Combine protein and ligand
        xtal = combine_protein_ligand(
            xtal, mol, resid=new_resid, start_atom_id=new_atomid
        )

        out_base = f"{compound_id}_{struct}"
        if args.out_dir:
            out_fn = f"{args.out_dir}/{out_base}_bound.pdb"
        else:
            out_fn = f"{os.path.dirname(docked_fn)}/{out_base}_bound.pdb"
        save_openeye_pdb(xtal, out_fn)

        print(f"Wrote {out_fn}", flush=True)


if __name__ == "__main__":
    main()
