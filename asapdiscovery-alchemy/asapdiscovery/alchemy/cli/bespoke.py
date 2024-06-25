from typing import Optional
import shutil
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
    type=click.Choice(["aimnet2", "mace", "xtb"])
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
    from rich import pretty
    from asapdiscovery.alchemy.cli.utils import print_header, print_message
    from asapdiscovery.alchemy.schema.fec import FreeEnergyCalculationNetwork
    from openff.bespokefit.workflows import BespokeWorkflowFactory
    from openff.bespokefit.executor.client import BespokeFitClient
    from openff.bespokefit.executor.services import current_settings
    from openff.toolkit import Molecule

    pretty.install()
    console = rich.get_console()
    print_header(console)

    client = BespokeFitClient(settings=current_settings())
    # make sure the client can be reached before we generate the bespokefit jobs
    # an error is raised if we can not hit the client
    _ = client.list_optimizations()

    if factory_file is not None:
        bespoke_factory = BespokeWorkflowFactory.from_file(file_name=factory_file)
        message = f"Loading BespokeWorkflowFactory from [repr.filename]{factory_file}[/repr.filename]"

    else:
        from openff.qcsubmit.common_structures import QCSpec
        # create a factory using pre-defined protocol
        # raise an error if we have charged ligands with the mace model
        protocols = {
            "aimnet2": QCSpec(method="wb97m-d3", basis=None, program="aimnet2", spec_description="ASAP-Alchemy standard aimnet2 protocol"),
            "mace": QCSpec(method="large", basis=None, program="mace", spec_description="ASAP-Alchemy standard mace protocol"),
            "xtb": QCSpec(method="gfn2xtb", basis=None, program="xtb", spec_description="ASAP-Alchemy standard xtb protocol")
        }
        bespoke_factory = BespokeWorkflowFactory(default_qc_specs=[protocols[protocol]])
        message = f"Creating BespokeWorkflowFactory from {protocol} protocol"

    print_message(console=console, message=message)

    fec_network = FreeEnergyCalculationNetwork.from_file(filename=network)
    print_message(console=console, message=f"Loaded {len(fec_network.network.ligands)} ligands from [repr.filename]{network}[/repr.filename]")

    bespoke_sub_status = console.status("Submitting BespokeFit jobs")
    bespoke_sub_status.start()
    # make sure the default force field matches our fec workflow
    # bespokefit will automatically strip the constraints if present
    bespoke_factory.initial_force_field = fec_network.forcefield_settings.small_molecule_forcefield

    submitted_ligands = []

    for ligand in fec_network.network.ligands:
        # create the job schema
        bespoke_job = bespoke_factory.optimization_schema_from_molecule(
            molecule=Molecule.from_rdkit(ligand.to_rdkit()),
            index=ligand.compound_name
        )
        # submit the job and save the task ID
        response = client.submit_optimization(input_schema=bespoke_job)
        ligand.tags["bespokefit_id"] = response
        submitted_ligands.append(ligand)

    # save the network back to file with the bespokefit ids
    fec_network.to_file(filename=network)

    bespoke_sub_status.stop()

    print_message(console=console, message=f"Saved FreeEnergyCalculationNetwork with BespokeFit ID's to [repr.filename]{network}[/repr.filename]")


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
    from rich import pretty
    from rich.progress import track
    from asapdiscovery.alchemy.cli.utils import print_header, print_message
    from asapdiscovery.alchemy.schema.fec import FreeEnergyCalculationNetwork
    from asapdiscovery.data.schema.identifiers import BespokeParameters, BespokeParameter
    from openff.bespokefit.executor.client import BespokeFitClient
    from openff.bespokefit.executor.services import current_settings

    pretty.install()
    console = rich.get_console()
    print_header(console)

    client = BespokeFitClient(settings=current_settings())

    fec_network = FreeEnergyCalculationNetwork.from_file(filename=network)
    print_message(console=console, message=f"Loaded {len(fec_network.network.ligands)} ligands from [repr.filename]{network}[/repr.filename]")

    # pull the optimisations from the server and store results if done
    for ligand in track(fec_network.network.ligands, description="Gathering bespoke parameters", console=console, transient=True, total=len(fec_network.network.ligands)):
        # make sure bespokefit ids are present
        if "bespokefit_id" in ligand.tags:
            bespoke_result = client.get_optimization(ligand.tags["bespokefit_id"])
            # we can only save the parameters if the optimisation has finished
            if bespoke_result.status == "success":

                refit_parameters = bespoke_result.results.refit_parameter_values
                # make sure we use the force field which we fit to
                # this was taken from the network originally
                bespoke_parameters = BespokeParameters(base_force_field=fec_network.forcefield_settings.small_molecule_forcefield)
                for parameter, values in refit_parameters.items():
                    bespoke_parameter = BespokeParameter(
                        interaction=parameter.type,
                        smirks=parameter.smirks,
                        values=dict((key, value.m) for key, value in values.items()),
                        units="kilocalories_per_mole"
                    )
                    bespoke_parameters.parameters.append(bespoke_parameter)
                # save the parameters back to the ligand
                ligand.bespoke_parameters = bespoke_parameters

            elif not allow_missing:
                raise RuntimeError(
                    "Not all BespokeFit optimisations have finished, to collect the current parameters use the flag "
                    "`--allow-missing`"
                )

    # write the updated network to the same file
    fec_network.to_file(filename=network)

    # workout how many ligands we updated
    success_opts = sum([1 for ligand in fec_network.network.ligands if ligand.bespoke_parameters is not None])

    print_message(console=console, message=f"Gathered bespoke parameters for {success_opts}/{len(fec_network.network.ligands)} ligands.")
    print_message(console=console, message=f"Saved FreeEnergyCalculationNetwork with BespokeFit parameters to [repr.filename]{network}[/repr.filename]")


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
    from rich import pretty
    from rich.progress import track
    from rich.table import Table
    from asapdiscovery.alchemy.schema.fec import FreeEnergyCalculationNetwork
    from asapdiscovery.alchemy.cli.utils import print_header, print_message
    from openff.bespokefit.executor.client import BespokeFitClient
    from openff.bespokefit.executor.services import current_settings

    pretty.install()
    console = rich.get_console()
    print_header(console)

    client = BespokeFitClient(settings=current_settings())

    fec_network = FreeEnergyCalculationNetwork.from_file(filename=network)

    print_message(console=console, message=f"Loaded {len(fec_network.network.ligands)} ligands from [repr.filename]{network}[/repr.filename]")

    table = Table()
    table.add_column("Status", justify="center", no_wrap=True)
    table.add_column("Count", overflow="fold")

    states = {"success": 0, "running": 0, "waiting": 0, "errored": 0}
    styles = ["green", "orange3", "#1793d0", "#ff073a"]

    for ligand in track(fec_network.network.ligands, description="Collecting status", total=len(fec_network.network.ligands), transient=True, console=console):
        response = client.get_optimization(ligand.tags["bespokefit_id"])
        status[response.status] += 1

    print_message(console=console, message="BespokeFit status breakdown")
    for (key, value), style in zip(states.items(), styles):
        table.add_row(key, str(value), style=style)
    console.print(table)
