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
    from rich.padding import Padding
    from asapdiscovery.alchemy.cli.utils import print_header, get_cpus
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
        protocols = {
            "aimnet2": QCSpec(method="wb97m-d3", basis=None, program="aimnet2", spec_description="ASAP-Alchemy standard aimnet2 protocol"),
            "mace": QCSpec(method="large", basis=None, program="mace", spec_description="ASAP-Alchemy standard mace protocol"),
            "xtb": QCSpec(method="gfn2xtb", basis=None, program="xtb", spec_description="ASAP-Alchemy standard xtb protocol")
        }
        bespoke_factory = BespokeWorkflowFactory(default_qc_specs=[protocols[protocol]])
        message = f"Creating BespokeWorkflowFactory from {protocol} protocol"

    console.print(Padding(message, (1, 0, 1, 0)))

    fec_network = FreeEnergyCalculationNetwork.from_file(filename=network)
    message = Padding(
        f"Loaded {len(fec_network.network.ligands)} ligands from [repr.filename]{network}[/repr.filename]",
        (1, 0, 1, 0)
    )
    console.print(message)

    bespoke_sub_status = console.status("Submitting BespokeFit jobs")
    bespoke_sub_status.start()
    # make sure the default force field matches our fec workflow
    # bespokefit will automatically strip the constraints if present
    bespoke_factory.initial_force_field = fec_network.forcefield_settings.small_molecule_forcefield

    # we should probably save the bespokefit protocol into the network to make sure its consistent for all future runs
    # this should be set in network planning?
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

    message = Padding(
        f"Saved FreeEnergyCalculationNetwork with BespokeFit ID's to [repr.filename]{network}[/repr.filename]",
        (1, 0, 1, 0),
    )

    console.print(message)







