import click
import itertools
from asapdiscovery.cli.cli_args import (
    dask_args,
    ligands,
    md_args,
    output_dir,
    pdb_file,
)
from asapdiscovery.simulation.simulate import VanillaMDSimulator

@click.group()
def simulation():
    """Run simulations on molecular systems."""
    pass


@simulation.command()
@ligands
@pdb_file
@md_args
@output_dir
def vanilla_md(
    
):
    print("Running vanilla MD simulation")
    simulator = VanillaMDSimulator()
    ligs = MolfileFactory.from_file(ligands).read()
    combo = list(itertools.product([pdb_file], ligs))
    print("running simulations ..., please wait")
    simulator.simulate(combo)
    print("done")



@simulation.command()
@ligands
@pdb_file
@md_args
def szybki():
    print("Running szybki simulation")