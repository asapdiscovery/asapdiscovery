import shutil
from typing import Optional

import click
from asapdiscovery.alchemy.cli.utils import SpecialHelpOrder


@click.group(
    cls=SpecialHelpOrder,
    context_settings={"max_content_width": shutil.get_terminal_size().columns - 20},
)
def bespoke():
    """Tools to generate bespoke torsion parameters using BespokeFit"""
    pass


@bespoke.command(
    short_help="Submit a set of ligands in a local FreeEnergyCalculationNetwork to a BespokeFit server.",
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
    "-f",
    "--factory-file",
    type=click.Path(resolve_path=True, exists=True, file_okay=True, dir_okay=False),
    help="The name of the JSON file containing the BespokeFit workflow, if not supplied the default protocol will be used.",
)
@click.option(
    "-p",
    "--protocol",
    default="aimnet2",
    show_default=True,
    help="The name of the predefined ASAP-Alchemy BespokeFit protocol to use.",
    type=click.Choice(["aimnet2", "mace", "xtb"]),
)
def submit(
    protocol: str,
    network: str,
    factory_file: Optional[str] = None,
):
    """
    Submit the ligands to a running BespokeFit server, we assume that relevant BespokeFit environment settings have
    been set.
    """
    import rich
    from asapdiscovery.alchemy.cli.utils import print_header, print_message
    from asapdiscovery.alchemy.schema.fec import FreeEnergyCalculationNetwork
    from asapdiscovery.alchemy.utils import BespokeFitHelper
    from openff.bespokefit.workflows import BespokeWorkflowFactory
    from rich import pretty

    pretty.install()
    console = rich.get_console()
    print_header(console)

    bespoke_client = BespokeFitHelper()
    # make sure the client can be reached before we generate the bespokefit jobs
    # an error is raised if we can not hit the client
    _ = bespoke_client._client.list_optimizations()

    if factory_file is not None:
        bespoke_factory = BespokeWorkflowFactory.from_file(file_name=factory_file)
        message = f"Loading BespokeWorkflowFactory from [repr.filename]{factory_file}[/repr.filename]"

    else:
        from openff.qcsubmit.common_structures import QCSpec

        # create a factory using pre-defined protocol
        # raise an error if we have charged ligands with the mace model
        protocols = {
            "aimnet2": QCSpec(
                method="wb97m-d3",
                basis=None,
                program="aimnet2",
                spec_description="ASAP-Alchemy standard aimnet2 protocol",
            ),
            "mace": QCSpec(
                method="large",
                basis=None,
                program="mace",
                spec_description="ASAP-Alchemy standard mace protocol",
            ),
            "xtb": QCSpec(
                method="gfn2xtb",
                basis=None,
                program="xtb",
                spec_description="ASAP-Alchemy standard xtb protocol",
            ),
        }
        bespoke_factory = BespokeWorkflowFactory(default_qc_specs=[protocols[protocol]])
        message = f"Creating BespokeWorkflowFactory from {protocol} protocol"

    print_message(console=console, message=message)

    fec_network = FreeEnergyCalculationNetwork.from_file(filename=network)
    print_message(
        console=console,
        message=f"Loaded {len(fec_network.network.ligands)} ligands from [repr.filename]{network}[/repr.filename]",
    )

    bespoke_sub_status = console.status("Submitting BespokeFit jobs")
    bespoke_sub_status.start()
    # make sure the default force field matches our fec workflow
    # bespokefit will automatically strip the constraints if present
    # we need to patch the file extension for bespopkefit
    ff_name = fec_network.protocol_settings.forcefield_settings.small_molecule_forcefield
    if ".offxml" not in ff_name:
        ff_name += ".offxml"

    bespoke_factory.initial_force_field = ff_name

    fec_network = bespoke_client.submit_ligands(
        network=fec_network, bespokefit_protocol=bespoke_factory
    )

    # save the network back to file with the bespokefit ids
    fec_network.to_file(filename=network)

    bespoke_sub_status.stop()

    print_message(
        console=console,
        message=f"Saved FreeEnergyCalculationNetwork with BespokeFit ID's to [repr.filename]{network}[/repr.filename]",
    )


@bespoke.command(
    short_help="Gather a set of bespoke parameters from a BespokeFit server for "
    "ligands in a local FreeEnergyCalculationNetwork.",
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
    "--allow-missing/--no-allow-missing",
    "allow_missing",
    default=False,
    help="If we should allow missing results when gathering from alchemiscale.",
)
def gather(network: str, allow_missing: bool):
    """
    Gather the bespoke parameters for the ligands in the network file from a BespokeFit server.

    Args:
        network: The name of the JSON file containing the FreeEnergyCalculationNetwork with the ligands
            we want parameters for
        allow_missing: If we should allow missing parameters.

    Raises:
        RuntimeError if not all optimisation are complete and allowing missing is False.
    """
    import rich
    from asapdiscovery.alchemy.cli.utils import print_header, print_message
    from asapdiscovery.alchemy.schema.fec import FreeEnergyCalculationNetwork
    from asapdiscovery.alchemy.utils import BespokeFitHelper
    from rich import pretty

    pretty.install()
    console = rich.get_console()
    print_header(console)

    bespoke_helper = BespokeFitHelper()

    fec_network = FreeEnergyCalculationNetwork.from_file(filename=network)
    print_message(
        console=console,
        message=f"Loaded {len(fec_network.network.ligands)} ligands from [repr.filename]{network}[/repr.filename]",
    )

    # pull the optimisations from the server and store results if done
    gather_status = console.status("Gathering bespoke parameters")
    gather_status.start()
    fec_network = bespoke_helper.gather_results(network=fec_network)
    gather_status.stop()

    bespoke_status = [
        ligand.bespoke_parameters is None for ligand in fec_network.network.ligands
    ]
    if all(bespoke_status):
        print_message(console=console, message="No bespoke optimizations found.")

    # workout if we have missing data
    elif not allow_missing and any(
        [ligand.bespoke_parameters is None for ligand in fec_network.network.ligands]
    ):
        raise RuntimeError(
            "Not all BespokeFit optimisations have finished, to collect the current parameters use the flag "
            "`--allow-missing`"
        )

    else:
        # write the updated network to the same file
        fec_network.to_file(filename=network)

        # workout how many ligands we updated
        success_opts = sum(
            [
                1
                for ligand in fec_network.network.ligands
                if ligand.bespoke_parameters is not None
            ]
        )

        print_message(
            console=console,
            message=f"Gathered bespoke parameters for {success_opts}/{len(fec_network.network.ligands)} ligands.",
        )
        print_message(
            console=console,
            message=f"Saved FreeEnergyCalculationNetwork with BespokeFit parameters to [repr.filename]{network}[/repr.filename]",
        )


@bespoke.command(
    short_help="Check the status of the BespokeFit optimisations for this network.",
)
@click.option(
    "-n",
    "--network",
    type=click.Path(resolve_path=True, readable=True, file_okay=True, dir_okay=False),
    help="The name of the JSON file containing a planned FEC network.",
    default="planned_network.json",
    show_default=True,
)
def status(network: str):
    """
    Check the progress of the BespokeFit optimisations for this network
    """
    import rich
    from asapdiscovery.alchemy.cli.utils import print_header, print_message
    from asapdiscovery.alchemy.schema.fec import FreeEnergyCalculationNetwork
    from asapdiscovery.alchemy.utils import BespokeFitHelper
    from rich import pretty
    from rich.table import Table

    pretty.install()
    console = rich.get_console()
    print_header(console)

    bespoke_helper = BespokeFitHelper()

    fec_network = FreeEnergyCalculationNetwork.from_file(filename=network)

    print_message(
        console=console,
        message=f"Loaded {len(fec_network.network.ligands)} ligands from [repr.filename]{network}[/repr.filename]",
    )

    query_status = console.status("Collecting status")
    query_status.start()
    states = bespoke_helper.status(network=fec_network)
    query_status.stop()

    if not any(list(states.values())):
        print_message(console=console, message="No bespokefit optimizations found")

    else:
        print_message(console=console, message="BespokeFit status breakdown")
        styles = ["green", "orange3", "#1793d0", "#ff073a"]
        table = Table()
        table.add_column("Status", justify="center", no_wrap=True)
        table.add_column("Count", overflow="fold")
        for (key, value), style in zip(states.items(), styles):
            table.add_row(key, str(value), style=style)
        console.print(table)
