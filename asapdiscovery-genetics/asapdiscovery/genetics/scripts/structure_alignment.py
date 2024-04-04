import argparse
from pathlib import Path

import pandas as pd
from asapdiscovery.genetics.calculate_rmsd import (
    save_alignment_pymol,
    select_best_colabfold,
)

parser = argparse.ArgumentParser(
    description="Align PDB structures generated from ColabFold"
)

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
    required=False,
    default="aligned_pdbs/",
    help="Path directory to save PDB of aligned target protein",
)

parser.add_argument(
    "--results-dir",
    type=str,
    required=False,
    default="./",
    help="Path to folder where all ColabFold results are stored.",
)

parser.add_argument(
    "--out-dir",
    type=str,
    required=False,
    default="./",
    help="Path to output folder given on ColabFold run.",
)

parser.add_argument(
    "--pymol-save",
    type=str,
    required=False,
    default="aligned_proteins.pse",
    help="Path to save pymol session with aligned proteins",
)

parser.add_argument(
    "--cf-format",
    type=str,
    required=False,
    default="*_unrelaxed_rank_001_alphafold2_ptm_model_1_seed_*.pdb",
    help="Format of pdb file saved by ColabFold. Will rarely be different than default. ",
)


def main():
    args = parser.parse_args()
    # check all the required files exist
    seq_file = Path(args.seq_file)
    if not seq_file.exists():
        raise FileNotFoundError(f"Sequence file {seq_file} does not exist")
    query_name = seq_file.stem

    ref_pdb = Path(args.ref_pdb)
    if not ref_pdb.exists():
        raise FileNotFoundError(f"Ref PDB file {ref_pdb} does not exist")

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(
            f"The folder with ColabFold results {results_dir} does not exist"
        )
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    pymol_save = args.pymol_save
    cf_format = args.cf_format

    aligned_pdbs = []
    seq_labels = []
    seq_df = pd.read_csv(seq_file)
    for index, row in seq_df.iterrows():
        # iterate over each csv entry
        mol = row["id"]
        cf_results = results_dir / query_name / mol / args.out_dir
        final_pdb = save_dir / f"{mol}_aligned.pdb"
        # Select best seed repetition
        min_rmsd, min_file = select_best_colabfold(
            cf_results,
            mol,
            ref_pdb,
            chain="A",
            final_pdb=final_pdb,
            default_CF=cf_format,
        )

        aligned_pdbs.append(min_file)
        seq_labels.append(mol)

    session_save = save_dir / pymol_save
    save_alignment_pymol(aligned_pdbs, seq_labels, ref_pdb, session_save)

    return


if __name__ == "__main__":
    main()
