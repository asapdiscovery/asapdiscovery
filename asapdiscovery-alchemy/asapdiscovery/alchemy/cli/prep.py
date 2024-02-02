from typing import Optional

import click


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
    import rich
    from asapdiscovery.alchemy.cli.utils import print_header
    from asapdiscovery.alchemy.schema.prep_workflow import AlchemyPrepWorkflow
    from rich import pretty
    from rich.padding import Padding

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
    help="The name of the JSON file which contains the prepared receptor complex including the crystal ligand.",
)
@click.option(
    "-cs",
    "--core-smarts",
    type=click.STRING,
    help="The SMARTS which should be used to select which atoms to constrain to the reference structure.",
)
@click.option(
    "-p",
    "--processors",
    default="auto",
    show_default=True,
    help="The number of processors which can be used to run the workflow in parallel. `auto` will use (all_cpus -1), `all` will use all"
    "or the exact number of cpus to use can be provided.",
)
def run(
    dataset_name: str,
    ligands: str,
    receptor_complex: str,
    factory_file: Optional[str] = None,
    core_smarts: Optional[str] = None,
    processors: int = 1,
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
    processors: The number of processors which can be used to run the workflow in parallel. `auto` will use all
    cpus -1, `all` will use all or the exact number of cpus to use can be provided.
    """
    import pathlib
    from multiprocessing import cpu_count

    import pandas
    import rich
    from asapdiscovery.alchemy.cli.utils import print_header
    from asapdiscovery.alchemy.schema.prep_workflow import AlchemyPrepWorkflow
    from asapdiscovery.data.schema_v2.complex import PreppedComplex
    from asapdiscovery.data.schema_v2.molfile import MolFileFactory
    from rich import pretty
    from rich.padding import Padding

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
    asap_ligands = MolFileFactory(filename=ligands).load()

    message = Padding(
        f"Loaded {len(asap_ligands)} ligands from [repr.filename]{ligands}[/repr.filename]",
        (1, 0, 1, 0),
    )
    console.print(message)
    # always expect the JSON file
    ref_complex = PreppedComplex.parse_file(receptor_complex)

    message = Padding(
        f"Loaded a prepared complex from [repr.filename]{receptor_complex}[/repr.filename]",
        (1, 0, 1, 0),
    )
    console.print(message)

    # workout the number of processes to use if auto or all
    all_cpus = cpu_count()
    if processors == "all":
        processors = all_cpus
    elif processors == "auto":
        processors = all_cpus - 1
    else:
        # can be a string from click
        processors = int(processors)

    message = Padding(
        f"Starting Alchemy-Prep workflow with {processors} processors", (1, 0, 1, 0)
    )
    console.print(message)

    alchemy_dataset = factory.create_alchemy_dataset(
        dataset_name=dataset_name,
        ligands=asap_ligands,
        reference_complex=ref_complex,
        processors=processors,
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
    receptor_file = output_folder.joinpath(
        f"{alchemy_dataset.reference_complex.target.target_name}.pdb"
    )
    alchemy_dataset.reference_complex.target.to_pdb_file(receptor_file)
    message = Padding(
        f"Saved receptor to [repr.filename]{receptor_file}[/repr.filename]",
        (1, 0, 1, 0),
    )
    console.print(message)

    # create a csv of the failed ligands with the failure reason
    if alchemy_dataset.failed_ligands:
        rows = []
        failed_ligands_file = output_folder.joinpath("failed_ligands.csv")
        message = Padding(
            f"[yellow]WARNING some ligands failed to have poses generated see [repr.filename]{failed_ligands_file}[/repr.filename][/yellow]",
            (1, 0, 1, 0),
        )
        console.print(message)
        for fail_type, ligands in alchemy_dataset.failed_ligands.items():
            for ligand in ligands:
                rows.append(
                    {
                        "smiles": ligand.provenance.isomeric_smiles,
                        "name": ligand.compound_name,
                        "failure type": fail_type,
                        # if it was an omega fail print the return code
                        "failure info": ligand.tags.get("omega_return_code", ""),
                    }
                )

        # write to csv
        df = pandas.DataFrame(rows)
        failed_ligands_file = output_folder.joinpath("failed_ligands.csv")
        df.to_csv(failed_ligands_file, index=False)
