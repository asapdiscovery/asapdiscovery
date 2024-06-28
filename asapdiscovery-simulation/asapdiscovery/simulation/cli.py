import click
import itertools
import logging 
from typing import Optional, Union
from pathlib import Path

from asapdiscovery.cli.cli_args import (
    use_dask,
    ligands,
    md_steps,
    md_openmm_platform,
    output_dir,
    pdb_file,
    loglevel,
)
from asapdiscovery.simulation.simulate import OpenMMPlatform, VanillaMDSimulator
from asapdiscovery.data.util.dask_utils import DaskType, make_dask_client_meta, BackendType
from asapdiscovery.data.util.logging import FileLogger
from asapdiscovery.data.readers.molfile import MolFileFactory

@click.group()
def simulation():
    """Run simulations on molecular systems."""
    pass


@simulation.command()
@ligands
@pdb_file
@md_steps
@md_openmm_platform
@output_dir
@loglevel   
@use_dask
def vanilla_md(
    ligands: Optional[str] = None,
    pdb_file: Optional[str] = None,
    md_steps: int = 2500000,  # 10 ns @ 4.0 fs timestep
    md_openmm_platform: OpenMMPlatform = OpenMMPlatform.Fastest,
    output_dir: str = "vanilla_md_output",
    loglevel: Union[int, str] = logging.INFO,
    use_dask: bool = False
    
):
    logger.info("Running vanilla MD simulation")
    
    logger = FileLogger(
        "",  # default root logger so that dask logging is forwarded
        path=output_dir,
        logfile="vanilla_md.log",
        stdout=True,
        level=loglevel,
    ).getLogger()

    if use_dask:
        logger.info("Using dask")
        dask_client = make_dask_client_meta(
                DaskType.LOCAL_GPU, loglevel=loglevel
            )
    else:
        dask_client = None

    pdb_path = Path(pdb_file)
    
    complex = Complex.from_pdb(
                pdb_file,
                target_kwargs={"target_name": pdb_path.stem},
                ligand_kwargs={"compound_name": f"{pdb_path.stem}_ligand"},
            )


    if ligands:
        # in case its a multisdf
        ligs = MolFileFactory.from_file(ligands).read()
        logger.info(f"Read {len(ligs)} ligands")

        # save each ligand to a file
        lig_paths = []
        lig_dir = output_dir / "ligands"
        for lig in ligs:
            path = lig_dir / f"{lig.compound_name}.sdf"
            lig.to_sdf(path)
            lig_paths.append(path)

    else:
        # use the ligand that was in the pdb
        if complex.ligand:
            lig_path = output_dir / f"{pdb_path.stem}_ligand.sdf"
            complex.ligand.to_sdf(lig_path)
            lig_paths = [lig_path]
        else:
            raise ValueError("No ligands provided and none found in pdb")
    
    protein_processed_path = output_dir / f"{pdb_path.stem}_processed.pdb"
    complex.to_pdb(protein_processed_path)

    combo = list(itertools.product([protein_processed_path], lig_paths))

    simulator = VanillaMDSimulator(output_dir=output_dir, openmm_platform=md_openmm_platform, num_steps=md_steps)

    logger.info("running simulations ..., please wait")
    simulator.simulate(combo, use_dask=use_dask, dask_client=dask_client)
    logger.info("done")



@simulation.command()
@ligands
@pdb_file
def szybki():
    raise NotImplementedError("Szybki simulation not yet implemented")
