import pathlib
from typing import Optional

import click
import rich
from rich import pretty
from rich.padding import Padding

from asapdiscovery.alchemy.cli.utils import print_header
from asapdiscovery.alchemy.schema.prep_workflow import AlchemyPrepWorkflow


@click.group()
def prep():
    """Tools to prepare ligands for Alchemy networks via state expansion and constrained pose generation."""


@prep.command()
@click.option(
    "-f",
    "--filename",
    type=click.Path(exists=False, file_okay=True, dir_okay=False, writable=True),
    help="The name of the JSON file the workflow should be saved to.",
    required=True,
)
@click.option(
    "-cs",
    "--core-smarts",
    type=click.STRING,
    help="The SMARTS which should be used to select which atoms to constrain to the reference structure.",
)
def create(filename: str, core_smarts: str):
    """
    Create a new AlchemyPrepWorkflow with default settings and save it to JSON file.
    """

    pretty.install()
    console = rich.get_console()
    print_header(console)
    factory = AlchemyPrepWorkflow(core_smarts=core_smarts)
    factory.to_file(filename=filename)
    message = Padding(
        f"The AlchemyPrepWorkflow has been saved to [repr.filename]{filename}[/repr.filename]",
        (1, 0, 1, 0),
    )
    console.print(message)


@prep.command()
@click.option(
    "-f",
    "--factory-file",
    type=click.Path(resolve_path=True, exists=True, file_okay=True, dir_okay=False),
    help="The name of the JSON file containing the AlchemyPrepWorkflow factory, if not supplied the default will be used.",
)
@click.option(
    "-n",
    "--dataset-name",
    type=click.STRING,
    help="The name of the AlchemyDataset this will also be the name of the folder created.",
)
@click.option(
    "-l",
    "--ligands",
    type=click.Path(resolve_path=True, exists=True, file_okay=True, dir_okay=False),
    help="The file which contains the ligands to use in the planned network.",
)
@click.option(
    "-r",
    "--receptor-complex",
    type=click.Path(resolve_path=True, exists=True, file_okay=True, dir_okay=False),
    help="The name of the file which contains the prepared receptor complex including the crystal ligand.",
)
@click.option(
    "-cs",
    "--core-smarts",
    type=click.STRING,
    help="The SMARTS string used to identify the atoms in each ligand which should be constrained to the reference pose.",
)
def run(
    dataset_name: str,
    ligands: str,
    receptor_complex: str,
    factory_file: Optional[str] = None,
    core_smarts: Optional[str] = None,
):
    """
    Create an AlchemyDataset by running the given AlchemyPrepWorkflow which will expand the ligand states and generate
    constrained poses suitable for ASAP-Alchemy.

    Parameters
    ----------
    dataset_name: The name which should be given to the AlchemyDataset all results will be saved in a folder with the same name.
    ligands: The name of the local file which contains the input ligands to be prepared in the workflow.
    receptor_complex: The name of the local file which contains the prepared complex including the crystal ligand.
    factory_file: The name of the JSON file with the configured AlchemyPrepWorkflow, if not supplied the default will be
        used but a core smarts must be provided.
    core_smarts: The SMARTS string used to identify the atoms in each ligand to be constrained. Required if the factory file is not supplied.
    """
    from asapdiscovery.data.openeye import save_openeye_sdfs
    from asapdiscovery.data.schema_v2.complex import PreppedComplex
    from asapdiscovery.data.schema_v2.molfile import MolFileFactory

    pretty.install()
    console = rich.get_console()
    print_header(console)

    # load the factory and set the core smarts if supplied
    if factory_file is not None:
        factory = AlchemyPrepWorkflow.parse_file(factory_file)
        if core_smarts is not None:
            factory.core_smarts = core_smarts
    else:
        factory = AlchemyPrepWorkflow(core_smarts=core_smarts)

    # load the molecules
    asap_ligands = MolFileFactory.from_file(filename=ligands).ligands

    message = Padding(
        f"Loaded {len(asap_ligands)} ligands from [repr.filename]{ligands}[/repr.filename]",
        (1, 0, 1, 0),
    )
    console.print(message)

    ref_complex = PreppedComplex.parse_file(receptor_complex)
    message = Padding(
        f"Loaded a prepared complex from [repr.filename]{receptor_complex}[/repr.filename]",
        (1, 0, 1, 0),
    )
    console.print(message)

    message = Padding("Starting Alchemy-Prep workflow", (1, 0, 1, 0))
    console.print(message)

    alchemy_dataset = factory.create_alchemy_dataset(
        dataset_name=dataset_name, ligands=asap_ligands, reference_complex=ref_complex
    )
    output_folder = pathlib.Path(dataset_name)
    output_folder.mkdir(parents=True, exist_ok=True)

    dataset_file = output_folder.joinpath("prepared_alchemy_dataset.json")
    alchemy_dataset.to_file(dataset_file)
    message = Padding(
        f"Saved AlchemyDataset to [repr.filename]{dataset_file}[/repr.filename]",
        (1, 0, 1, 0),
    )
    console.print(message)

    posed_ligand_file = output_folder.joinpath("posed_ligands.sdf")
    alchemy_dataset.save_posed_ligands(posed_ligand_file)
    message = Padding(
        f"Saved posed ligands to [repr.filename]{posed_ligand_file}[/repr.filename]",
        (1, 0, 1, 0),
    )
    console.print(message)

    if alchemy_dataset.failed_ligands:
        message = Padding(
            f"[yellow]WARNING some ligands failed to have poses generated see failed_ligands files in [repr.filename]{output_folder}[/repr.filename][/yellow]",
            (1, 0, 1, 0),
        )
        console.print(message)
    for fail_type, ligands in alchemy_dataset.failed_ligands.items():
        fails = [ligand.to_oemol() for ligand in ligands]
        failed_ligand_file = output_folder.joinpath(f"failed_ligands_{fail_type}.sdf")
        save_openeye_sdfs(fails, failed_ligand_file)
