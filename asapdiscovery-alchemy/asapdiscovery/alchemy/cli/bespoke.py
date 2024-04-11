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
    short_help="Submit a set of ligands in a local FreeEnergyCalculationNetwork to a bespokefit server."
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
    choices=click.Choice(["aimnet2", "mace", "xtb"])
)
@click.option(
    "-pr",
    "--processors",
    default="auto",
    show_default=True,
    help="The number of processors which can be used to run the workflow in parallel. `auto` will use (all_cpus -1), "
    "`all` will use all or the exact number of cpus to use can be provided.",
)
def submit(
    protocol: str,
    network: str,
    processors: str | int,
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
    from asapdiscovery.alchemy.interfaces import BespokeFitSettings
    from asapdiscovery.alchemy.schema.fec import FreeEnergyCalculationNetwork
    from openff.bespokefit.workflows import BespokeWorkflowFactory
    from openff.toolkit import Molecule

    pretty.install()
    console = rich.get_console()
    print_header(console)

    settings = BespokeFitSettings()

    if factory_file is not None:
        bespoke_factory = BespokeWorkflowFactory.from_file(file_name=factory_file)
        message = f"Loading BespokeWorkflowFactory from [repr.filename]{factory_file}[/repr.filename]"

    else:
        from openff.qcsubmit.common_structures import QCSpec
        # create a factory using a protocol
        protocols = {
            "aimnet2": QCSpec(method="wb97m-d3", basis=None, program="aimnet2", spec_description="ASAP-Alchemy standard protocol"),
            "mace": QCSpec(method="large", basis=None, program="mace", spec_description="ASAP-Alchemy standard protocol"),
            "xtb": QCSpec(method="gfn2xtb", basis=None, program="xtb", spec_description="ASAP-Alchemy standard protocol")
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

    # make sure the client can be reached before we generate the bespokefit jobs

    bespoke_prep_status = console.status("Generating BespokeFit jobs")
    bespoke_prep_status.start()
    # make sure the default force field matches our fec workflow
    bespoke_factory.initial_force_field = fec_network.forcefield_settings.small_molecule_forcefield
    # we should probably save the bespokefit protocol into the network to make sure its consistent for all future runs
    # this should be set in network planning?

    # workout how many processors to use
    processors = get_cpus(processors)
    bespoke_jobs = bespoke_factory.optimization_schemas_from_molecules(
        molecules=[Molecule.from_rdkit(ligand.to_rdkit(), allow_undefined_stereo=True) for ligand in fec_network.network.ligands],
        processors=processors
    )








