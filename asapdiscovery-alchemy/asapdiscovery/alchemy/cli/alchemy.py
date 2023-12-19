from typing import Optional

import click


@click.group()
def alchemy():
    """Tools to create and execute Alchemy networks using OpenFE and alchemiscale."""
    pass


@alchemy.command()
@click.argument(
    "filename",
    type=click.Path(exists=False, file_okay=True, dir_okay=False, writable=True),
)
def create(filename: str):
    """
    Create a new free energy perturbation factory with default settings and save it to JSON file.

    Args:
        filename: The name of the JSON file containing the factory schema.
    """
    from asapdiscovery.alchemy.schema.fec import FreeEnergyCalculationFactory

    factory = FreeEnergyCalculationFactory()
    factory.to_file(filename=filename)


@alchemy.command()
@click.option(
    "-f",
    "--factory-file",
    type=click.Path(resolve_path=True, exists=True, file_okay=True, dir_okay=False),
    help="The name of the JSON file containing the FEC factory, if not supplied the default will be used.",
)
@click.option(
    "-n",
    "--name",
    type=click.STRING,
    help="The name which should be given to this dataset.",
)
@click.option(
    "-r",
    "--receptor",
    type=click.Path(resolve_path=True, exists=True, file_okay=True, dir_okay=False),
    help="The name of the file which contains the prepared receptor.",
)
@click.option(
    "-l",
    "--ligands",
    type=click.Path(resolve_path=True, exists=True, file_okay=True, dir_okay=False),
    help="The file which contains the ligands to use in the planned network.",
)
@click.option(
    "-ad",
    "--alchemy-dataset",
    type=click.Path(resolve_path=True, exists=True, file_okay=True, dir_okay=False),
    help="The JSON file containing an AlchemyDataset created with ASAP-Alchemy prep run. This defines the ligands and the receptor.",
)
@click.option(
    "-c",
    "--center-ligand",
    type=click.Path(resolve_path=True, exists=True, file_okay=True, dir_okay=False),
    help="The file which contains the center ligand, only required by radial type networks.",
)
def plan(
    name: str,
    receptor: Optional[str] = None,
    ligands: Optional[str] = None,
    center_ligand: Optional[str] = None,
    factory_file: Optional[str] = None,
    alchemy_dataset: Optional[str] = None,
):
    """
    Plan a FreeEnergyCalculationNetwork using the given factory and inputs. The planned network will be written to file
    in a folder named after the dataset.
    """
    import pathlib

    import openfe
    from asapdiscovery.alchemy.schema.fec import FreeEnergyCalculationFactory
    from asapdiscovery.alchemy.schema.prep_workflow import AlchemyDataSet
    from rdkit import Chem

    # check mutually exclusive args
    if ligands is None and alchemy_dataset is None:
        raise RuntimeError(
            "Please provide either an AlchemyDataSet created with `asap-alchemy prep run` or ligand and receptor input files."
        )

    click.echo("Loading FreeEnergyCalculationFactory ...")
    # parse the factory is supplied else get the default
    if factory_file is not None:
        factory = FreeEnergyCalculationFactory.from_file(factory_file)

    else:
        factory = FreeEnergyCalculationFactory()

    if alchemy_dataset is not None:
        import tempfile

        # load the set of posed ligands and the receptor from our dataset
        click.echo(f"Loading Ligands and protein from AlchemyDataSet {alchemy_dataset}")
        alchemy_ds = AlchemyDataSet.from_file(alchemy_dataset)
        input_ligands = [
            openfe.SmallMoleculeComponent.from_sdf_string(mol.to_sdf_str())
            for mol in alchemy_ds.posed_ligands
        ]
        # write to a temp pdb file and read back in
        with tempfile.NamedTemporaryFile(suffix=".pdb") as fp:
            alchemy_ds.reference_complex.target.to_pdb_file(fp.name)
            receptor = openfe.ProteinComponent.from_pdb_file(fp.name)

    else:
        # load from separate files
        click.echo(f"Loading Ligands from {ligands}")
        # parse all required data/ assume sdf currently
        supplier = Chem.SDMolSupplier(ligands, removeHs=False)
        input_ligands = [
            openfe.SmallMoleculeComponent.from_rdkit(mol) for mol in supplier
        ]
        click.echo(f"Loading protein from {receptor}")
        receptor = openfe.ProteinComponent.from_pdb_file(receptor)

    if center_ligand is not None:
        # handle the center ligand needed for radial networks
        supplier = Chem.SDMolSupplier(center_ligand, removeHs=False)
        center_ligand = [
            openfe.SmallMoleculeComponent.from_rdkit(mol) for mol in supplier
        ]
        if len(center_ligand) > 1:
            raise RuntimeError(
                f"Only a single center ligand can be used for radial networks, found {len(center_ligand)} ligands in {center_ligand}."
            )

        center_ligand = center_ligand[0]

    click.echo("Creating FEC network ...")
    planned_network = factory.create_fec_dataset(
        dataset_name=name,
        receptor=receptor,
        ligands=input_ligands,
        central_ligand=center_ligand,
    )
    click.echo(f"Writing results to {name}")
    # output the data to a folder named after the dataset
    output_folder = pathlib.Path(name)
    output_folder.mkdir(parents=True, exist_ok=True)

    network_file = output_folder.joinpath("planned_network.json")
    planned_network.to_file(network_file)

    graph_file = output_folder.joinpath("ligand_network.graphml")
    with graph_file.open("w") as output:
        output.write(planned_network.network.graphml)


@alchemy.command()
@click.option(
    "-n",
    "--network",
    type=click.Path(resolve_path=True, readable=True, file_okay=True, dir_okay=False),
    help="The name of the JSON file containing a planned FEC network.",
    default="planned_network.json",
    show_default=True,
)
@click.option(
    "-o",
    "--organization",
    type=click.STRING,
    default="asap",
    help="The name of the organization in alchemiscale the network should be submitted to.",
    show_default=True,
)
@click.option(
    "-c",
    "--campaign",
    type=click.STRING,
    help="The name of the campaign in alchemiscale the network should be submitted to.",
    required=True,
)
@click.option(
    "-p",
    "--project",
    type=click.STRING,
    help="The name of the project in alchemiscale the network should be submitted to.",
    required=True,
)
def submit(network: str, organization: str, campaign: str, project: str):
    """
    Submit a local FreeEnergyCalculationNetwork to alchemiscale using the provided scope details. The network object
    will have these details saved into it.

    Args:
        network: The name of the JSON file containing the FreeEnergyCalculation to be submitted.
        organization: The name of the organization this network should be submitted under always asap.
        campaign: The name of the campaign this network should be submitted under.
        project: The name of the project this network should be submitted under.
    """
    from alchemiscale import Scope
    from asapdiscovery.alchemy.schema.fec import FreeEnergyCalculationNetwork
    from asapdiscovery.alchemy.utils import AlchemiscaleHelper

    # launch the helper which will try to login
    click.echo("Connecting to Alchemiscale...")
    client = AlchemiscaleHelper()
    # create the scope
    network_scope = Scope(org=organization, campaign=campaign, project=project)
    # load the network
    planned_network = FreeEnergyCalculationNetwork.from_file(network)
    # create network on alchemiscale
    click.echo(
        f"Creating network on Alchemiscale instance: {client._client.api_url} with scope {network_scope}"
    )
    submitted_network = client.create_network(
        planned_network=planned_network, scope=network_scope
    )
    # write the network with its key to file before we try and add compute incase we hit an issue
    click.echo("Network made; saving network key to network file")
    submitted_network.to_file(network)
    # now action the tasks
    click.echo("Creating and actioning FEC tasks on Alchemiscale...")
    task_ids = client.action_network(planned_network=submitted_network)
    # check that all tasks were created
    missing_tasks = sum([1 for task in task_ids if task is None])
    total_tasks = len(task_ids)
    click.echo(
        f"{total_tasks - missing_tasks}/{total_tasks} created. Status can be checked using `asap-alchemy status`"
    )


@alchemy.command()
@click.option(
    "-n",
    "--network",
    type=click.Path(resolve_path=True, readable=True, file_okay=True, dir_okay=False),
    help="The name of the JSON file containing a planned FEC network.",
    default="planned_network.json",
    show_default=True,
)
@click.option(
    "--allow-missing/--no-allow-missing",
    "allow_missing",
    default=False,
    help="If we should allow missing results when gathering from alchemiscale.",
)
def gather(network: str, allow_missing: bool):
    """
    Gather the results from alchemiscale for the given network.

    Note: An error is raised if all calculations have not finished and allow-missing is False.

    Args:
        network: The of the JSON file containing the FreeEnergyCalculationNetwork whos results we should gather.
        allow_missing: If we should allow missing results when trying to gather the network.

    Raises:
        Runtime error if all calculations are not complete and allow missing is False.
    """
    from asapdiscovery.alchemy.schema.fec import FreeEnergyCalculationNetwork
    from asapdiscovery.alchemy.utils import AlchemiscaleHelper

    # launch the helper which will try to login
    click.echo("Connecting to Alchemiscale...")
    client = AlchemiscaleHelper()

    # load the network
    planned_network = FreeEnergyCalculationNetwork.from_file(network)

    # show the network status
    status = client.network_status(planned_network=planned_network)
    if not allow_missing and "waiting" in status:
        raise RuntimeError(
            "Not all calculations have finished, to collect the current results use the flag `--allow_missing`."
        )

    click.echo(
        f"Gathering network results from Alchemiscale instance: {client._client.api_url} with key {planned_network.results.network_key}"
    )
    network_with_results = client.collect_results(planned_network=planned_network)
    click.echo("Results gathered saving to file ...")
    network_with_results.to_file("result_network.json")


@alchemy.command()
@click.option(
    "-n",
    "--network",
    type=click.Path(resolve_path=True, readable=True, file_okay=True, dir_okay=False),
    help="The name of the JSON file containing a planned FEC network.",
    default="planned_network.json",
    show_default=True,
)
@click.option(
    "-e",
    "--errors",
    is_flag=True,
    default=False,
    help="Output errors from the network, if any.",
)
@click.option(
    "-t",
    "--with-traceback",
    is_flag=True,
    default=False,
    help="Output the errors and tracebacks from the failing tasks.",
)
def status(network: str, errors: bool, with_traceback: bool):
    """
    Get the status of the submitted network on alchemiscale.\f

    Args:
        network: The name of the JSON file containing the FreeEnergyCalculationNetwork we should check the status of.
        errors: Flag to show errors from the tasks.
        with_traceback: Flag to show the complete traceback for the errored tasks.

    """
    from asapdiscovery.alchemy.schema.fec import FreeEnergyCalculationNetwork
    from asapdiscovery.alchemy.utils import AlchemiscaleHelper

    # launch the helper which will try to login
    client = AlchemiscaleHelper()
    # load the network
    planned_network = FreeEnergyCalculationNetwork.from_file(network)
    # check the status
    client.network_status(planned_network=planned_network)
    # collect errors
    if errors or with_traceback:
        task_errors = client.collect_errors(
            planned_network,
        )
        # output errors in readable format
        for failure in task_errors:
            click.echo(click.style("Task:", bold=True))
            click.echo(f"{failure.task_key}")
            click.echo(click.style("Error:", bold=True))
            click.echo(f"{failure.error}")
            if with_traceback:
                click.echo(click.style("Traceback:", bold=True))
                click.echo(f"{failure.traceback}")
            click.echo()


@alchemy.command()
@click.option(
    "-n",
    "--network",
    type=click.Path(resolve_path=True, readable=True, file_okay=True, dir_okay=False),
    help="The name of the JSON file containing a planned FEC network.",
    default="planned_network.json",
    show_default=True,
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Increase verbosity of output; will give ScopedKeys of restarted Tasks",
)
@click.argument(
    "tasks",
    nargs=-1,
)
def restart(network: str, verbose: bool, tasks):
    """Restart errored Tasks for the given FEC network.

    If TASKS specified, then only these will be restarted.

    """
    from alchemiscale import ScopedKey
    from asapdiscovery.alchemy.schema.fec import FreeEnergyCalculationNetwork
    from asapdiscovery.alchemy.utils import AlchemiscaleHelper

    client = AlchemiscaleHelper()
    planned_network = FreeEnergyCalculationNetwork.from_file(network)

    tasks = [ScopedKey.from_str(task) for task in tasks]

    restarted_tasks = client.restart_tasks(planned_network, tasks)
    if verbose:
        click.echo(f"Restarted Tasks: {[str(i) for i in restarted_tasks]}")
    else:
        click.echo(f"Restarted {len(restarted_tasks)} Tasks")


@alchemy.command()
@click.option(
    "-n",
    "--network",
    type=click.Path(resolve_path=True, readable=True, file_okay=True, dir_okay=False),
    help="The name of the JSON file containing a planned FEC network with raw results from alchemiscale.",
    default="result_network.json",
    show_default=True,
)
@click.option(
    "-rd",
    "--reference-dataset",
    type=click.Path(resolve_path=True, readable=True, file_okay=True, dir_okay=False),
    help="The name of a csv file containing reference experimental data to be used in the predictions."
)
@click.option(
    "-ru",
    "--reference-units",
    type=click.Choice(["pIC50", "IC50"]),
    help="The units of the reference experimental data provided in the csv or saved as an SDTag on the ligand.",
    default="pIC50",
    show_default=True
)
def predict(network: str, reference_units: str, reference_dataset: Optional[str] = None):
    """
    Predict relative and absolute free energies for the set of ligands, using any provided experimental data to shift the
    results to the relevant energy range.

    """
    from asapdiscovery.alchemy.schema.fec import FreeEnergyCalculationNetwork
    from asapdiscovery.alchemy.predict import get_data_from_femap, create_absolute_report, create_relative_report

    result_network = FreeEnergyCalculationNetwork.from_file(network)
    ligands = result_network.network.to_openfe_ligands()
    # convert to cinnabar fepmap to do the prediction via MLE
    fe_map = result_network.results.to_fe_map()
    fe_map.generate_absolute_values()
    absolute_df, relative_df = get_data_from_femap(
        fe_map=fe_map,
        ligands=ligands,
        assay_units=reference_units,
        reference_dataset=reference_dataset
    )
    # write the csv to file to be uploaded to postera later
    absolute_df.to_csv("absolute-predictions.csv")
    relative_df.to_csv("relative-predictions.csv")

    absolute_layout = create_absolute_report(dataframe=absolute_df)
    absolute_layout.save("Absolute-prediction.html", title="ASAP-Alchemy-Absolute", embed=True)

    relative_layout = create_relative_report(dataframe=relative_df)
    relative_layout.save("Relative-prediction.html", title="ASAP-Alchemy-Relative", embed=True)






