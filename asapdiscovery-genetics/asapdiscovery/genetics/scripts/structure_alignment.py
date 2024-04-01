from pathlib import Path
from asapdiscovery.genetics.calculate_rmsd import select_best_colabfold, save_alignment_pymol

import argparse
parser = argparse.ArgumentParser(description="Align PDB structures generated from ColabFold")

parser.add_argument(
    "--seq-file",
    type=str,
    required=True,
    help="Path to the csv with the sequences.", 
)
parser.add_argument(
    "--ref-pdb",
    type=str,
    required=True,
    help="Path to PDB file of reference structure.",
)

parser.add_argument(
    "--save-dir",
    type=str,
    required=True,
    help="Path directory to save PDB of aligned target protein",
)

parser.add_argument(
    "--results-CF",
    type=str,
    required=True,
    help="Path to folder where ColabFold results are stored",
)

parser.add_argument(
    "--pymol-save",
    type=str,
    required=False,
    default="pymol_alignment.pse",
    help="Path to save pymol session with aligned proteins",
)

def main():
    args = parser.parse_args()
    # check all the required files exist
    seq_file = Path(args.seq_file)
    if not seq_file.exists():
        raise FileNotFoundError(f"Sequence file {seq_file} does not exist")

    ref_pdb = Path(args.ref_pdb)
    if not ref_pdb.exists():
        raise FileNotFoundError(f"Ref PDB file {ref_pdb} does not exist")

    results_dir = Path(args.results_CF)
    if not results_dir.exists():
        raise FileNotFoundError(
            f"The folder with ColabFold results {results_dir} does not exist"
            )   
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
   
    pymol_save = args.pymol_save

    aligned_pdbs = []
    with open(seq_file, "r") as f:
        # Skip the first line
        next(f)
        for line in f:
            # Remove spaces
            line = line.strip("\n").lstrip("\r")
            mol, seq = line.split(",", 1) 
            cf_results = results_dir / args.seq_file.split(".")[0] / mol / "results"
            final_pdb = save_dir / f"{mol}_aligned.pdb"

            min_rmsd, min_file = select_best_colabfold(cf_results, 
                                                       mol, ref_pdb, 
                                                       chain="A", 
                                                       final_pdb=final_pdb)

            aligned_pdbs.append(min_file)

    save_alignment_pymol(aligned_pdbs, pymol_save)

    return

if __name__ == "__main__":
    main()
