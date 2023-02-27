import argparse

import pandas
from asapdiscovery.data.openeye import oechem, oedocking, oegrid
from asapdiscovery.data.openeye import load_openeye_pdb, split_openeye_mol


SCORE_TYPES = {
    "chemgauss": oedocking.OEScoreType_Chemgauss4,
    "chemgauss3": oedocking.OEScoreType_Chemgauss3,
    "chemgauss4": oedocking.OEScoreType_Chemgauss4,
    "chemscore": oedocking.OEScoreType_Chemscore,
    "shapegauss": oedocking.OEScoreType_Shapegauss,
    "plp": oedocking.OEScoreType_PLP,
}


def get_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-s", "--structure_fn", required=True, help="Protein structure file."
    )
    parser.add_argument("-l", "--ligand_fn", required=True, help="SDF ligand file.")
    parser.add_argument("-o", "--out_fn", required=True, help="Output CSV file.")

    parser.add_argument(
        "-c", "--score_type", nargs="*", help="Which scoring type(s) to use."
    )

    parser.add_argument(
        "-g",
        "--grid_base",
        help=(
            "Grid base name (must end in .ccp4 and have an empty {} for " "formatting)."
        ),
    )

    return parser.parse_args()


def main():
    args = get_args()

    # Load the protein (and just protein)
    prot = split_openeye_mol(load_openeye_pdb(args.structure_fn))["pro"]

    # Set up ligand mol stream
    ifs = oechem.oemolistream()
    ifs.open(args.ligand_fn)

    # Sanitize score types
    if not args.score_type:
        # Don't use chemgauss key because it's a duplicate
        use_scores = list(SCORE_TYPES.keys())[1:]
    else:
        use_scores = [s.lower() for s in args.score_type if s.lower() in SCORE_TYPES]
    if len(use_scores) == 0:
        raise RuntimeError("No valid score types")
    use_types = [SCORE_TYPES[s] for s in use_scores]

    all_scores = []
    for i, lig in enumerate(ifs.GetOEGraphMols()):
        # Get ligand coordinates
        lig_coords = [c for xyz in lig.GetCoords().values() for c in xyz]

        # Build ligand box (extend the box just to get everything)
        box = oechem.OEBox()
        box.Setup(
            oechem.FloatArray(lig_coords),
            lig.NumAtoms(),
        )
        box.Extend(5.0)

        # Save grid
        if args.grid_base:
            minmax = oechem.FloatArray(
                [
                    box.GetXMin(),
                    box.GetYMin(),
                    box.GetZMin(),
                    box.GetXMax(),
                    box.GetYMax(),
                    box.GetZMax(),
                ]
            )
            g = oegrid.OEScalarGrid(minmax, 0.5)
            oegrid.OEWriteGrid(args.grid_base.format(i), g)

        tmp_scores = []
        for st in use_types:
            # Set up scorer
            scorer = oedocking.OEScore(st)
            # scorer = oedocking.OEScore(oedocking.OEScoreType_Chemgauss4)
            scorer.Initialize(prot, box)
            lig_score = scorer.ScoreLigand(lig)
            tmp_scores.append(lig_score)
        all_scores.append(tmp_scores)

    df = pandas.DataFrame(all_scores, columns=use_scores)
    df.to_csv(args.out_fn)


if __name__ == "__main__":
    main()
