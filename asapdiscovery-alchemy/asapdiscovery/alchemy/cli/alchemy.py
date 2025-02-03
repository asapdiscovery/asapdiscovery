import shutil
from pathlib import Path
from typing import Optional

import click
from asapdiscovery.alchemy.cli.utils import SpecialHelpOrder
from asapdiscovery.data.services.postera.manifold_data_validation import (
    TagEnumBase,
    TargetTags,
)


@click.group(
    cls=SpecialHelpOrder,
    context_settings={"max_content_width": shutil.get_terminal_size().columns - 20},
)
def alchemy():
    """Tools to create and execute Alchemy networks using OpenFE and alchemiscale."""
    pass


@alchemy.command(
    help_priority=1,
    short_help="Create a new free energy perturbation factory with default settings and save it to JSON file.",
)
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


@alchemy.command(
    help_priority=2,
    short_help="Plan a FreeEnergyCalculationNetwork using the given factory and inputs. The planned network will be written to file in a folder named after the dataset.",
)
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
@click.option(
    "-g",
    "--graphml",
    help="Read a graphml representation of the ligand network directly from file",
    type=click.Path(resolve_path=True, exists=True, file_okay=True, dir_okay=False),
)
@click.option(
    "-cn",
    "--custom-network-file",
    type=click.Path(resolve_path=True, exists=True, file_okay=True, dir_okay=False),
    help="An optional path to a custom network specified as a CSV file where each line contains <lig_a,lig_b>, on the next line <lig_b,lig_x>, etc.",
)
@click.option(
    "-ep",
    "--experimental-protocol",
    help="The name of the experimental protocol in the CDD vault that should be associated with this Alchemy network.",
    type=click.STRING,
    default=None,
    show_default=True,
)
@click.option(
    "-t",
    "--target",
    help="The name of the biological target associated with this workflow.",
    type=click.Choice(TargetTags.get_values(), case_sensitive=True),
)
def plan(
    name: Optional[str] = None,
    receptor: Optional[str] = None,
    ligands: Optional[str] = None,
    center_ligand: Optional[str] = None,
    graphml: Optional[str] = None,
    custom_network_file: Optional[str] = None,
    factory_file: Optional[str] = None,
    alchemy_dataset: Optional[str] = None,
    experimental_protocol: Optional[str] = None,
    target: Optional[TagEnumBase] = None,
):
    """
    Plan a FreeEnergyCalculationNetwork using the given factory and inputs. The planned network will be written to file
    in a folder named after the dataset.
    """
    import pathlib

    import openfe
    from asapdiscovery.alchemy.schema.fec import FreeEnergyCalculationFactory
    from asapdiscovery.alchemy.schema.prep_workflow import AlchemyDataSet
    from asapdiscovery.data.readers.molfile import MolFileFactory

    # check mutually exclusive args
    if ligands and graphml:
        raise RuntimeError(
            "Please provide either a ligand file or a graphml file, not both."
        )

    if graphml and custom_network_file:
        raise RuntimeError(
            "Please provide either a graphml file or a custom network file, not both."
        )

    # nothing specified
    if ligands is None and graphml is None and alchemy_dataset is None:
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

        if graphml:
            raise RuntimeError(
                "Please provide either dataset file or a graphml file, not both."
            )
        # load the set of posed ligands and the receptor from our dataset
        click.echo(f"Loading Ligands and protein from AlchemyDataSet {alchemy_dataset}")
        alchemy_ds = AlchemyDataSet.from_file(alchemy_dataset)
        input_ligands = alchemy_ds.posed_ligands

        # workout which name should be used, the CLI input takes priority over the prep `dataset_name`.
        name = name or alchemy_ds.dataset_name

        # write to a temp pdb file and read back in
        with tempfile.NamedTemporaryFile(suffix=".pdb") as fp:
            alchemy_ds.reference_complex.target.to_pdb_file(fp.name)
            receptor = openfe.ProteinComponent.from_pdb_file(fp.name)

    else:
        if graphml:
            # load from graphml further down the line
            click.echo("Loading Ligands from graphml ...")
            input_ligands = None
        else:
            # load from separate files
            click.echo(f"Loading Ligands from {ligands}")
            # parse all required data/ assume sdf currently
            input_ligands = MolFileFactory(filename=ligands).load()

        click.echo(f"Loading protein from {receptor}")
        receptor = openfe.ProteinComponent.from_pdb_file(receptor)

    if center_ligand is not None:
        # handle the center ligand needed for radial networks
        center_ligand = MolFileFactory(filename=center_ligand).load()
        if len(center_ligand) > 1:
            raise RuntimeError(
                f"Only a single center ligand can be used for radial networks, found {len(center_ligand)} ligands in {center_ligand}."
            )

        center_ligand = center_ligand[0]

    if graphml is not None:
        # load the graphml file
        with open(graphml) as f:
            graphml = f.read()
        click.echo("Graphml file loaded: Using explicit ligand network.")

    if not name:
        raise RuntimeError("Please provide a name for the dataset.")

    if custom_network_file is not None:
        from asapdiscovery.alchemy.schema.network import CustomNetworkPlanner
        from asapdiscovery.alchemy.utils import extract_custom_ligand_network

        click.echo(
            f"Using custom network specified in {custom_network_file}, ignoring network mapper settings and central ligand if supplied."
        )
        factory.network_planner.network_planning_method = CustomNetworkPlanner(
            edges=extract_custom_ligand_network(custom_network_file)
        )
    click.echo("Creating FEC network ...")
    planned_network = factory.create_fec_dataset(
        dataset_name=name,
        receptor=receptor,
        ligands=input_ligands,
        central_ligand=center_ligand,
        experimental_protocol=experimental_protocol,
        target=target,
        graphml=graphml,
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


@alchemy.command(
    help_priority=3,
    short_help="Submit a local FreeEnergyCalculationNetwork to alchemiscale using the provided scope details. The network object will have these details saved into it.",
)
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
    help="The name of the campaign in alchemiscale the network should be submitted to. If `-o` is set to 'asap' (default), `-c` must be either of 'public' or 'confidential'.",
    required=True,
)
@click.option(
    "-p",
    "--project",
    type=click.STRING,
    help="The name of the project in alchemiscale the network should be submitted to.",
    required=True,
)
@click.option(
    "-r",
    "--repeats",
    type=click.INT,
    help="The total number of times each transformation should be ran on Alchemiscale, results will be averaged over "
    "the successful repeats.",
    default=1,
    show_default=True,
)
def submit(
    network: str,
    organization: str,
    campaign: str,
    project: str,
    repeats: int,
):
    """
    Submit a local FreeEnergyCalculationNetwork to alchemiscale using the provided scope details. The network object
    will have these details saved into it.

    Args:
        network: The name of the JSON file containing the FreeEnergyCalculation to be submitted.
        organization: The name of the organization this network should be submitted under always asap.
        campaign: The name of the campaign this network should be submitted under.
        project: The name of the project this network should be submitted under.
        repeats: The total number of times each transformation should be ran.
    """
    import rich
    from alchemiscale import Scope
    from asapdiscovery.alchemy.cli.utils import print_header, print_message
    from asapdiscovery.alchemy.schema.fec import FreeEnergyCalculationNetwork
    from asapdiscovery.alchemy.utils import AlchemiscaleHelper
    from rich import pretty

    pretty.install()
    console = rich.get_console()
    print_header(console)

    # make sure the org/campaign combination is valid
    if organization == "asap" and campaign not in ("public", "confidential"):
        raise ValueError(
            "If organization (`-o`) is set to 'asap' (default), campaign (`-c`) must be either of 'public' or 'confidential'."
        )

    # launch the helper which will try to login
    print_message(console=console, message="Connecting to Alchemiscale")
    client = AlchemiscaleHelper.from_settings()
    # create the scope
    network_scope = Scope(org=organization, campaign=campaign, project=project)
    # load the network
    planned_network = FreeEnergyCalculationNetwork.from_file(network)
    # create network on alchemiscale
    print_message(
        console=console,
        message=(
            f"Creating network on Alchemiscale instance: {client._client.api_url} with scope {network_scope}"
        ),
    )
    submitted_network = client.create_network(
        planned_network=planned_network, scope=network_scope
    )

    # write the network with its key to file before we try and add compute incase we hit an issue
    print_message(
        console=console, message="Network made; saving network key to network file"
    )
    submitted_network.to_file(network)
    # now action the tasks
    print_message(
        console=console, message="Creating and actioning FEC tasks on Alchemiscale"
    )
    task_ids = client.action_network(planned_network=submitted_network, repeats=repeats)
    # check that all tasks were created
    missing_tasks = sum([1 for task in task_ids if task is None])
    total_tasks = len(task_ids)
    print_message(
        console=console,
        message=f"{total_tasks - missing_tasks}/{total_tasks} created. Status can be checked using `asap-alchemy status`",
    )


@alchemy.command(
    help_priority=7,
    short_help="Gather the results from alchemiscale for the given network.",
)
@click.option(
    "-n",
    "--network",
    type=click.Path(resolve_path=True, readable=True, file_okay=True, dir_okay=False),
    help="The name of the JSON file containing a submitted FEC network (typically 'planned_network.json').",
    default="planned_network.json",
    show_default=True,
)
@click.option(
    "-nk",
    "--network_key",
    type=click.STRING,
    help="The network key of a submitted FEC network.",
    default=None,
    show_default=True,
)
@click.option(
    "--allow-missing/--no-allow-missing",
    "allow_missing",
    default=False,
    help="If we should allow missing results when gathering from alchemiscale.",
)
def gather(
    network: Optional[str] = False,
    network_key: Optional[str] = False,
    allow_missing: Optional[bool] = None,
):
    """
    Gather the results from alchemiscale for the given network.

    Note: An error is raised if all calculations have not finished and allow-missing is False.

    Args:
        network: The path of the JSON file containing the FreeEnergyCalculationNetwork whose results we should gather.
        network_key: The `alchemsicale` network key of the network whose results we should gather.
        allow_missing: If we should allow missing results when trying to gather the network.

    Raises:
        Runtime error if all calculations are not complete and allow missing is False.
    """
    from asapdiscovery.alchemy.schema.fec import FreeEnergyCalculationNetwork
    from asapdiscovery.alchemy.utils import AlchemiscaleHelper

    # launch the helper which will try to login
    click.echo("Connecting to Alchemiscale...")
    client = AlchemiscaleHelper.from_settings()

    # load the network
    using_network_key = False
    if network_key:
        if Path(network).exists():
            click.echo(
                f"Network key provided: {network_key}, prefering over network file {network}."
            )
        using_network_key = True
    else:
        click.echo(f"Network file provided: {network}, loading network.")
        if not Path(network).exists():
            raise FileNotFoundError(f"Network file {network} does not exist.")

        planned_network = FreeEnergyCalculationNetwork.from_file(network)
        network_key = planned_network.results.network_key

    # check network key exists in alchemiscale
    try:
        if not client.network_exists(network_key=network_key):
            raise ValueError(
                f"Network key {network_key} does not exist in Alchemiscale."
            )
    except Exception as e:
        raise ValueError(
            f"Network key {network_key} does not exist in Alchemiscale."
        ) from e

    # show the network status
    status = client.network_status(network_key=network_key)
    if not allow_missing and "waiting" in status:
        raise RuntimeError(
            "Not all calculations have finished, to collect the current results use the flag `--allow-missing`."
        )

    click.echo(
        f"Gathering network results from Alchemiscale instance: {client._client.api_url} with key {network_key}"
    )

    if using_network_key:
        network_with_results = client.collect_results(network_key=network_key)
    else:
        network_with_results = client.collect_results(planned_network=planned_network)

    click.echo("Results gathered saving to file ...")
    network_with_results.to_file("result_network.json")


@alchemy.command(
    help_priority=4,
    short_help="Get the status of the submitted network on alchemiscale.",
)
@click.option(
    "-n",
    "--network",
    type=click.Path(resolve_path=True, readable=True, file_okay=True, dir_okay=False),
    help="The name of the JSON file containing a planned FEC network.",
    default="planned_network.json",
    show_default=True,
    required=False,
)
@click.option(
    "-nk",
    "--network_key",
    type=click.STRING,
    help="The network key of the network to get the status of.",
    default=None,
    required=False,
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
@click.option(
    "-a",
    "--all-networks",
    is_flag=True,
    default=False,
    help="If the status of all running tasks in your scope should be displayed. "
    "This option will cause the command to ignore all other flags.",
)
def status(
    network: str,
    network_key: str,
    errors: bool,
    with_traceback: bool,
    all_networks: bool,
):
    """
    Get the status of the submitted network on alchemiscale.\f

    Args:
        network: The name of the JSON file containing the FreeEnergyCalculationNetwork we should check the status of.
        network_key: The network key of the network to get the status of.
        errors: Flag to show errors from the tasks.
        with_traceback: Flag to show the complete traceback for the errored tasks.
        all_networks: If that status of all networks under the users scope should be displayed rather than for a single network.

    Notes:
        The `all_networks` flag will ignore all other flags, to get error details for a specific network use the network argument.

    """
    import rich
    from asapdiscovery.alchemy.cli.utils import print_header
    from asapdiscovery.alchemy.schema.fec import FreeEnergyCalculationNetwork
    from asapdiscovery.alchemy.utils import AlchemiscaleHelper
    from rich import pretty
    from rich.table import Table

    pretty.install()
    console = rich.get_console()
    print_header(console)

    args = [all_networks, network_key]

    if sum([bool(arg) for arg in args]) > 1:
        raise ValueError(
            "Can not retrieve status for --network_key at the same time as --all-networks  Please flag only one of --network_key, --all-networks and --network_key"
        )
    # launch the helper which will try to login
    client = AlchemiscaleHelper.from_settings()
    if all_networks:
        click.echo("Getting status of all networks in scope")
        # show the results of all tasks in scope, this will print to terminal
        client._client.get_scope_status()

        # now get the status break down for each network in scope with running or waiting tasks only
        running_networks = client._client.query_networks()

        status_breakdown = console.status("Creating status breakdown")
        status_breakdown.start()

        # format into a rich table
        table = Table()
        table.add_column("Network Key", justify="center", no_wrap=True)
        table.add_column(
            "Complete", overflow="fold", style="green", header_style="green"
        )
        table.add_column(
            "Running", overflow="fold", style="orange3", header_style="orange3"
        )
        table.add_column(
            "Waiting", overflow="fold", style="#1793d0", header_style="#1793d0"
        )
        table.add_column(
            "Error", overflow="fold", style="#ff073a", header_style="#ff073a"
        )
        table.add_column(
            "Invalid", overflow="fold", style="magenta1", header_style="magenta1"
        )
        table.add_column(
            "Deleted", overflow="fold", style="purple", header_style="purple"
        )
        table.add_column(
            "Actioned", overflow="fold", style="orange_red1", header_style="orange_red1"
        )
        table.add_column(
            "Priority",
            overflow="fold",
            style="dark_turquoise",
            header_style="dark_turquoise",
        )

        networks_status = client._client.get_networks_status(networks=running_networks)
        networks_actioned_tasks = client._client.get_networks_actioned_tasks(
            networks=running_networks
        )
        network_weights = client._client.get_networks_weight(networks=running_networks)

        # sort the networks by weight so that we get the ones with highest weights showing first in the table
        networks_data = zip(
            running_networks, networks_status, networks_actioned_tasks, network_weights
        )

        for key, network_status, actioned_tasks, network_weight in sorted(
            networks_data, key=lambda element: element[-1], reverse=True
        ):
            if (
                "running" in network_status or "waiting" in network_status
            ) and actioned_tasks:
                table.add_row(
                    str(key),
                    str(network_status.get("complete", 0)),
                    str(network_status.get("running", 0)),
                    str(network_status.get("waiting", 0)),
                    str(network_status.get("error", 0)),
                    str(network_status.get("invalid", 0)),
                    str(network_status.get("deleted", 0)),
                    str(len(actioned_tasks)),
                    str(network_weight),
                )
        status_breakdown.stop()
        console.print(table)
    else:
        if network_key:
            if Path(network).exists():
                click.echo(
                    f"Network key provided: {network_key}, prefering over network file {network}."
                )

        else:
            click.echo(f"Network file provided: {network}, loading network.")
            if not Path(network).exists():
                raise FileNotFoundError(f"Network file {network} does not exist.")
            planned_network = FreeEnergyCalculationNetwork.from_file(network)
            network_key = planned_network.results.network_key

        # check the status
        try:
            if not client.network_exists(network_key=network_key):
                raise ValueError(
                    f"Network key {network_key} does not exist in Alchemiscale."
                )
        except Exception as e:
            raise ValueError(
                f"Network key {network_key} does not exist in Alchemiscale."
            ) from e

        client.network_status(network_key=network_key)

        # collect errors
        if errors or with_traceback:
            task_errors = client.collect_errors(network_key=network_key)

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


@alchemy.command(
    help_priority=5, short_help="Restart errored Tasks for the given FEC network."
)
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

    client = AlchemiscaleHelper.from_settings()
    planned_network = FreeEnergyCalculationNetwork.from_file(network)

    tasks = [ScopedKey.from_str(task) for task in tasks]

    restarted_tasks = client.restart_tasks(planned_network, tasks)
    if verbose:
        click.echo(f"Restarted Tasks: {[str(i) for i in restarted_tasks]}")
    else:
        click.echo(f"Restarted {len(restarted_tasks)} Tasks")


@alchemy.command(
    help_priority=6,
    short_help="Adjust a network's priority. The scheduler picks tasks to action by weight, if this network's weight"
    + " is set to 0.99 it will be picked in 99% of queries if there is one other network that has a weight of 0.01.",
)
@click.option(
    "-nk",
    "--network-key",
    type=click.STRING,
    help="The network key of the network to be stopped. This can be found by running e.g. `asap-alchemy status -a`.",
    required=True,
)
@click.option(
    "-w",
    "--weight",
    type=click.FloatRange(min=0.0, max=1.0),
    help="The weight that should be assigned to the network. Network weights can be found by running"
    + " `asap-alchemy status -a`.",
    required=True,
)
def prioritize(network_key: str, weight: float):
    """Adjust a network's weight to influence how often its tasks will be actioned compared to other networks."""
    import rich
    from asapdiscovery.alchemy.cli.utils import print_header
    from asapdiscovery.alchemy.utils import AlchemiscaleHelper
    from rich import pretty
    from rich.padding import Padding

    pretty.install()
    console = rich.get_console()
    print_header(console)

    client = AlchemiscaleHelper.from_settings()
    adjust_weight_status = console.status(
        f"Changing weight of network {network_key} to {weight}"
    )
    adjust_weight_status.start()
    new_weight, old_weight = client.adjust_weight(
        network_key=network_key, weight=weight
    )
    adjust_weight_status.stop()

    # verify that the weight has been changed
    if not new_weight == weight:
        raise ValueError(
            f"Something went wrong during the weight change of network {network_key}:\nAttempted weight change "
            f"to {weight} but weight is {new_weight}."
        )

    message = Padding(
        f"Adjusted weight from {old_weight} to {new_weight} for network {network_key}",
        (1, 0, 1, 0),
    )
    console.print(message)


@alchemy.command(
    help_priority=7,
    short_help="Stop (i.e. set to 'error') a network's running and waiting tasks.",
)
@click.option(
    "-nk",
    "--network-key",
    type=click.STRING,
    help="The network key of the network to be stopped. This can be found by running e.g. `asap-alchemy status -a`.",
    required=True,
)
@click.option(
    "--hard",
    is_flag=True,
    help="If used, all waiting and running tasks will be deleted instead of un-actioned. Warning: these tasks will not be retrievable/re-runnable.",
)
def stop(network_key: str, hard: bool = False):
    """Stop (i.e. set to 'error') a network's running and waiting tasks."""
    import rich
    from asapdiscovery.alchemy.cli.utils import print_header
    from asapdiscovery.alchemy.utils import AlchemiscaleHelper
    from rich import pretty
    from rich.padding import Padding

    pretty.install()
    console = rich.get_console()
    print_header(console)

    client = AlchemiscaleHelper.from_settings()

    verb = "Deleted" if hard else "Canceled"

    if hard:
        console.print(
            f"Warning: deleting all running/waiting tasks on network {network_key}. These will not be retrievable/re-runnable!"
        )
        inp = input("Continue? (y/n)")
        if inp == "y":
            pass
        elif inp == "n":
            print("Aborting.")
            return
        else:
            raise ValueError("Option not recognized.")
    cancel_status = console.status(f"Canceling actioned tasks on network {network_key}")
    cancel_status.start()
    canceled_tasks = client.cancel_actioned_tasks(network_key=network_key, hard=hard)
    # check how many were canceled as some maybe None if not found
    total_tasks = len([task for task in canceled_tasks if task is not None])
    cancel_status.stop()

    message = Padding(
        f"{verb} {total_tasks} actioned tasks for network {network_key}",
        (1, 0, 1, 0),
    )
    console.print(message)


@alchemy.command(
    help_priority=8,
    short_help="Predict relative and absolute free energies for the set of ligands, using any provided experimental data to shift the results to the relevant energy range.",
)
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
    help="The name of a csv file containing reference experimental data to be used in the predictions.",
)
@click.option(
    "-ru",
    "--reference-units",
    type=click.Choice(["pIC50", "IC50"]),
    help="The units of the reference experimental data provided in the csv or saved as an SDTag on the ligand.",
    default="pIC50",
    show_default=True,
)
@click.option(
    "-ep",
    "--experimental-protocol",
    help="The name of the experimental protocol in the CDD vault that should be associated with this Alchemy network.",
    type=click.STRING,
    default=None,
    show_default=True,
)
@click.option(
    "-t",
    "--target",
    help="The name of the biological target associated with this workflow.",
    type=click.Choice(TargetTags.get_values(), case_sensitive=True),
)
@click.option(
    "-pm",
    "--postera-molset-name",
    type=click.STRING,
    default=None,
    show_default=True,
    help="The name of the Postera molecule set to upload the results to.",
)
@click.option(
    "-c",
    "--clean",
    is_flag=True,
    help="Whether or not to clean the incoming result network, e.g. in cases where some edges are imbalanced between complex/solvent or when DG==0.0.",
)
@click.option(
    "-fl",
    "--force-largest",
    is_flag=True,
    help="Make predictions using only the largest subnetwork present in the results. "
    "Useful in cases where the network is disconnected by e.g. simulation failures.",
)
@click.option(
    "-wtop",
    "--write-top-n-poses",
    help="The number of top-scoring poses to write to a multi-SDF in the local directory. By default writes the top 1000 (or all if the ligand series is smaller).",
    type=click.INT,
    default=1000,
    show_default=False,
)
def predict(
    network: str,
    reference_units: str,
    reference_dataset: Optional[str] = None,
    experimental_protocol: Optional[str] = None,
    target: Optional[TagEnumBase] = None,
    postera_molset_name: Optional[str] = None,
    clean: Optional[bool] = False,
    force_largest: Optional[bool] = False,
    write_top_n_poses: Optional[int] = 1000,
):
    """
    Predict relative and absolute free energies for the set of ligands, using any provided experimental data to shift the
    results to the relevant energy range.
    """
    import numpy as np
    import rich
    from asapdiscovery.alchemy.cli.utils import (
        cinnabar_femap_get_largest_subnetwork,
        cinnabar_femap_is_connected,
        print_header,
        upload_to_postera,
    )
    from asapdiscovery.alchemy.predict import (
        clean_result_network,
        create_absolute_report,
        create_relative_report,
        get_data_from_femap,
        get_top_n_poses,
    )
    from asapdiscovery.alchemy.schema.fec import FreeEnergyCalculationNetwork
    from rich import pretty
    from rich.padding import Padding

    pretty.install()
    console = rich.get_console()
    print_header(console)

    if clean:
        message = Padding(
            f"Warning: cleaning incoming result network {network}. You may lose results.",
            (1, 0, 1, 0),
        )
        console.print(message)
        result_network = clean_result_network(network, console=console)
    else:
        result_network = FreeEnergyCalculationNetwork.from_file(network)

    message = Padding(
        f"Loaded FreeEnergyCalculationNetwork from [repr.filename]{network}[/repr.filename]",
        (1, 0, 1, 0),
    )
    console.print(message)

    predict_status = console.status("Calculating absolute free energies")
    predict_status.start()

    # gather all ligands needed for the prediction labels
    ligands = result_network.network.ligands
    if result_network.network.central_ligand is not None:
        ligands.append(result_network.network.central_ligand)

    # convert to cinnabar fepmap to do the prediction via MLE
    fe_map = result_network.results.to_fe_map()
    is_connected = cinnabar_femap_is_connected(fe_map)

    if is_connected:
        try:
            fe_map.generate_absolute_values()
        except np.linalg.LinAlgError:
            raise ValueError(
                "MLE failed during absolute value generation. Does your result network contain "
                "NaNs? You can manually remove these or run `predict -c` to remove them automatically."
            )
    elif not is_connected and force_largest:
        fe_map = cinnabar_femap_get_largest_subnetwork(fe_map, result_network, console)
        fe_map.generate_absolute_values()
    else:
        raise ValueError(
            "Your network is missing edges resulting in a gap where ligands (nodes) are "
            "not connected to the network. If you would like to discard those disconnected ligands "
            "(i.e. not make predictions on them), run predict using the '-fl/--force-largest' flag."
        )

    # check if we have a protocol on the network already to draw experimental results from
    protocol = experimental_protocol or result_network.experimental_protocol

    absolute_df, relative_df = get_data_from_femap(
        fe_map=fe_map,
        ligands=ligands,
        assay_units=reference_units,
        reference_dataset=reference_dataset,
        cdd_protocol=protocol,
    )

    # write predictions to csv file
    absolute_path = f"predictions-absolute-{result_network.dataset_name}.csv"
    relative_path = f"predictions-relative-{result_network.dataset_name}.csv"
    absolute_df.to_csv(absolute_path)
    relative_df.to_csv(relative_path)
    predict_status.stop()
    message = Padding(
        f"Absolute predictions written to [repr.filename]{absolute_path}[/repr.filename]",
        (1, 0, 1, 0),
    )
    console.print(message)
    message = Padding(
        f"Relative predictions written to [repr.filename]{relative_path}[/repr.filename]",
        (1, 0, 1, 0),
    )
    console.print(message)

    # if requested, write an SDF of the top n compounds' docked poses
    if write_top_n_poses > 0:
        _ = get_top_n_poses(
            absolute_df, ligands, write_top_n_poses, console, write_file=True
        )

    # check if we have a biological target
    bio_target = target or result_network.target

    # workout if we should upload to postera
    if bio_target is not None and postera_molset_name is not None:
        # format and upload to postera
        postera_status = console.status(
            f"Uploading predictions to Postera Manifold molecule set: {postera_molset_name}."
        )
        postera_status.start()

        upload_to_postera(
            molecule_set_name=postera_molset_name,
            target=target,
            absolute_dg_predictions=absolute_df,
        )

        message = Padding(
            f"Predictions uploaded to Postera Manifold molecule set: {postera_molset_name}",
            (1, 0, 1, 0),
        )
        postera_status.stop()
        console.print(message)

    elif postera_molset_name is not None and bio_target is None:
        message = Padding(
            "[yellow]WARNING a postera molecule set name was provided without a target, results will not be uploaded! "
            "Please run again and provide a valid target `-t`[/yellow]",
            (1, 0, 1, 0),
        )
        console.print(message)

    # create interactive reports, they will work out if a plot should be included
    report_status = console.status("Generating interactive reports")
    report_status.start()
    # we can only make these reports currently with experimental data
    # TODO update once we have the per replicate estimate and error
    absolute_layout = create_absolute_report(dataframe=absolute_df)
    absolute_path = f"predictions-absolute-{result_network.dataset_name}.html"
    relative_path = f"predictions-relative-{result_network.dataset_name}.html"
    absolute_layout.save(
        absolute_path,
        title=f"ASAP-Alchemy-Absolute-{result_network.dataset_name}",
        embed=True,
    )

    relative_layout = create_relative_report(dataframe=relative_df)
    relative_layout.save(
        relative_path,
        title=f"ASAP-Alchemy-Relative-{result_network.dataset_name}",
        embed=True,
    )
    report_status.stop()

    message = Padding(
        f"Absolute report written to [repr.filename]{absolute_path}[/repr.filename]",
        (1, 0, 1, 0),
    )
    console.print(message)
    message = Padding(
        f"Relative report written to [repr.filename]{relative_path}[/repr.filename]",
        (1, 0, 1, 0),
    )
    console.print(message)
