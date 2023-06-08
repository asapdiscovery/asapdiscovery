import argparse
import multiprocessing as mp
from functools import partial
from pathlib import Path

from asapdiscovery.data.logging import FileLogger
from asapdiscovery.modeling.modeling import protein_prep_workflow
from asapdiscovery.modeling.schema import PrepOpts, PreppedTargets


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
        type=Path,
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
    parser.add_argument(
        "--spruce_only",
        action="store_true",
        default=False,
        help="If true, instead of generating design units only the spruced protein will be generated.",
    )

    # Performance and Debugging
    parser.add_argument(
        "-n",
        "--num_cores",
        type=int,
        default=1,
        help="Number of concurrent processes to run.",
    )
    parser.add_argument(
        "--debug_num",
        type=int,
        default=None,
        help="Number of targets to prep. Useful for debugging and testing.",
    )
    return parser.parse_args()


def main():
    args = get_args()
    args.output_dir.mkdir(exist_ok=True, parents=True)
    logger = FileLogger(
        logname="protein_prep_workflow", path=str(args.output_dir)
    ).getLogger()

    # Load the targets
    logger.info(f"Loading targets from {args.input_file}")
    targets: list = PreppedTargets.from_pkl(args.input_file).iterable
    logger.info(f"Loaded {len(targets)} targets")
    if args.debug_num is not None:
        targets = targets[: args.debug_num]
        logger.info(f"Only prepping {args.debug_num} targets because of --debug_num")
    prep_opts = PrepOpts(
        ref_fn=args.ref_prot,
        ref_chain=args.ref_chain,
        loop_db=args.loop_db,
        seqres_yaml=args.seqres_yaml,
        output_dir=args.output_dir,
        make_design_unit=not args.spruce_only,
    )
    logger.info(f"Prep options: {prep_opts}")

    # add prep opts to the prepping function
    protein_prep_workflow_with_opts = partial(
        protein_prep_workflow, prep_opts=prep_opts
    )

    # Run the prepping workflow
    logger.info(
        f"Running protein prep workflow on {len(targets)} targets using {args.num_cores} cores"
    )
    nprocs = min(mp.cpu_count(), len(targets), args.num_cores)
    with mp.Pool(processes=nprocs) as pool:
        targets_list = pool.map(protein_prep_workflow_with_opts, targets)

    # Write out the prepped targets
    output_pkl = args.output_dir / "prepped_targets.pkl"
    logger.info(f"Writing prepped targets to {output_pkl}")
    PreppedTargets.from_list(targets_list).to_pkl(output_pkl)
