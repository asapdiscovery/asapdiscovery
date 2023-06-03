from asapdiscovery.modeling.schema import PreppedTargets, PrepOpts
from asapdiscovery.modeling.modeling import protein_prep_workflow
import argparse
from pathlib import Path
import multiprocessing as mp
from functools import partial


def get_args():
    parser = argparse.ArgumentParser(description="")

    # Input/Output arguments
    parser.add_argument(
        "-i",
        "--input_file",
        type=Path,
        required=True,
        help="Path to input file containing the PreppedTargets",
    )
    # Output arguments
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="Path to output_dir.",
    )

    # Prep Options
    parser.add_argument(
        "-r",
        "--ref_prot",
        required=False,
        type=Path,
        help="Path to reference pdb to align to. If None, no alignment will be performed",
    )
    parser.add_argument(
        "--ref_chain", type=str, default="A", help="Chain of reference to align to."
    )
    parser.add_argument(
        "-l",
        "--loop_db",
        required=False,
        help="Path to loop database.",
    )
    parser.add_argument(
        "-s",
        "--seqres_yaml",
        required=False,
        help="Path to yaml file of SEQRES.",
    )

    # Performance and Debugging
    parser.add_argument(
        "-n",
        "--num_cores",
        type=int,
        default=1,
        help="Number of concurrent processes to run.",
    )
    return parser.parse_args()


def main():
    args = get_args()
    targets: list = PreppedTargets.from_pkl(args.input_file).iterable
    prep_opts = PrepOpts(
        ref_fn=args.ref_prot,
        ref_chain=args.ref_chain,
        loop_db=args.loop_db,
        seqres_yaml=args.seqres_yaml,
        output_dir=args.output_dir,
    )

    print(f"Running protein prep workflow on {len(targets)} targets using {prep_opts}")

    # add prep opts to the prepping function
    protein_prep_workflow_with_opts = partial(
        protein_prep_workflow, prep_opts=prep_opts
    )

    nprocs = min(mp.cpu_count(), len(targets), args.num_cores)
    with mp.Pool(processes=nprocs) as pool:
        targets_list = pool.map(protein_prep_workflow_with_opts, targets)

    # Write out the prepped targets
    PreppedTargets.from_list(targets_list).to_pkl(
        args.output_dir / "prepped_targets.pkl"
    )
