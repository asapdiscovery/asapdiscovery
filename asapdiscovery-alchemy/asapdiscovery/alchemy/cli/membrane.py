import shutil
from typing import Optional

import click

from asapdiscovery.alchemy.cli.utils import SpecialHelpOrder


@click.group(
    short_help="Build membrane-embedded receptors for membrane-bound alchemistry.",
    cls=SpecialHelpOrder,
    context_settings={"max_content_width": shutil.get_terminal_size().columns - 20},
)
def membrane():
    """Build membrane-embedded receptors for membrane-bound alchemistry.

    OpenFE's membrane workflow takes a fully built, solvated, pre-equilibrated system as a
    single PDB containing protein + lipids + water + ions and NO ligand, with explicit box
    vectors and aligned to the coordinate axes. These commands produce that file; feed it
    to `asap-alchemy plan --membrane-receptor`.
    """


def _load_spec(preset: Optional[str], spec_file: Optional[str]):
    from asapdiscovery.alchemy.membrane import MembraneAlignmentSpec, get_preset

    if spec_file and preset:
        raise click.ClickException("Provide either --preset or --spec-file, not both.")
    if spec_file:
        return MembraneAlignmentSpec.from_file(spec_file)
    if preset:
        try:
            return get_preset(preset)
        except KeyError as error:
            raise click.ClickException(str(error))
    raise click.ClickException(
        "A membrane alignment spec is required: pass --preset (e.g. 'herg') or "
        "--spec-file with a MembraneAlignmentSpec JSON."
    )


def _load_settings(settings_file: Optional[str]):
    from asapdiscovery.alchemy.membrane import MembraneBuildSettings

    if settings_file:
        return MembraneBuildSettings.from_file(settings_file)
    return MembraneBuildSettings()


@membrane.command(
    short_help="List the membrane alignment presets shipped with ASAP-Alchemy.",
    help_priority=1,
)
def presets():
    """List the membrane alignment presets shipped with ASAP-Alchemy."""
    import rich
    from rich.table import Table

    from asapdiscovery.alchemy.cli.utils import print_header
    from asapdiscovery.alchemy.membrane import PRESETS

    console = rich.get_console()
    print_header(console)

    table = Table()
    table.add_column("Preset", justify="left", no_wrap=True)
    table.add_column("Axis", justify="left")
    table.add_column("Midplane helices", justify="left")
    table.add_column("Description", overflow="fold")
    for name, spec in sorted(PRESETS.items()):
        table.add_row(
            name,
            f"{spec.axis_from.start}-{spec.axis_from.end} -> "
            f"{spec.axis_to.start}-{spec.axis_to.end}",
            ", ".join(spec.midplane_helices),
            spec.description,
        )
    console.print(table)


@membrane.command(
    short_help="Report the membrane geometry for a receptor without building anything.",
    help_priority=2,
)
@click.option(
    "-r",
    "--receptor",
    type=click.Path(resolve_path=True, exists=True, file_okay=True, dir_okay=False),
    required=True,
    help="The prepared receptor PDB.",
)
@click.option(
    "-p", "--preset", type=click.STRING, help="A shipped alignment preset, e.g. 'herg'."
)
@click.option(
    "-s",
    "--spec-file",
    type=click.Path(resolve_path=True, exists=True, file_okay=True, dir_okay=False),
    help="A MembraneAlignmentSpec JSON file, instead of --preset.",
)
def analyse(receptor: str, preset: Optional[str], spec_file: Optional[str]):
    """Report the membrane geometry for a receptor without building anything.

    Fast, and the right first step on a new structure: it tells you how far the axis was
    off z, where the midplane landed, and whether the landmark depths are consistent with
    the preset. If the landmarks fail here, they will fail after a multi-hour build too.
    """
    import json

    import rich
    from rich import pretty

    from asapdiscovery.alchemy.cli.utils import print_header
    from asapdiscovery.alchemy.membrane import align_receptor

    pretty.install()
    console = rich.get_console()
    print_header(console)

    spec = _load_spec(preset, spec_file)
    try:
        _, _, _, info = align_receptor(receptor, spec)
    except ValueError as error:
        raise click.ClickException(str(error))
    console.print(json.dumps(info, indent=1))


@membrane.command(
    short_help="Build, solvate and equilibrate a bilayer around a receptor.",
    help_priority=3,
)
@click.option(
    "-r",
    "--receptor",
    type=click.Path(resolve_path=True, exists=True, file_okay=True, dir_okay=False),
    required=True,
    help="The prepared receptor PDB (protein only, hydrogens added).",
)
@click.option(
    "-p", "--preset", type=click.STRING, help="A shipped alignment preset, e.g. 'herg'."
)
@click.option(
    "-s",
    "--spec-file",
    type=click.Path(resolve_path=True, exists=True, file_okay=True, dir_okay=False),
    help="A MembraneAlignmentSpec JSON file, instead of --preset.",
)
@click.option(
    "-o",
    "--output-prefix",
    type=click.Path(resolve_path=True),
    required=True,
    help="Prefix for the output files: <prefix>_membrane_system.pdb (the OpenFE input), "
    "<prefix>_complex_equil.pdb (with the pocket ligand, for inspection) and "
    "<prefix>_membrane_report.json.",
)
@click.option(
    "-e",
    "--equilibration-ns",
    type=click.FLOAT,
    default=20.0,
    show_default=True,
    help="Total equilibration length in ns, including the restrained stages. Bilayers "
    "need far longer than a soluble protein: a 2 ns hERG bilayer was still visibly "
    "drifting and produced a ~30%% task failure rate with SimulationNaNError, all of it "
    "in the complex leg.",
)
@click.option(
    "-lig",
    "--pocket-ligand",
    type=click.Path(resolve_path=True, exists=True, file_okay=True, dir_okay=False),
    help="An SDF ligand to hold the binding site open during solvation, stripped before "
    "the OpenFE input is written. Without it, waters land in the empty pocket and clash "
    "with whatever ligand is attached downstream. Use the crystallographic ligand.",
)
@click.option(
    "-set",
    "--settings-file",
    type=click.Path(resolve_path=True, exists=True, file_okay=True, dir_okay=False),
    help="A MembraneBuildSettings JSON file. Defaults reproduce the hERG production "
    "systems (POPC, 1.5 nm padding, 0.15 M NaCl, amber14 + lipid17).",
)
def build(
    receptor: str,
    preset: Optional[str],
    spec_file: Optional[str],
    output_prefix: str,
    equilibration_ns: float,
    pocket_ligand: Optional[str],
    settings_file: Optional[str],
):
    """Build, solvate and equilibrate a bilayer around a receptor.

    This runs an MD equilibration and wants a GPU; expect tens of minutes to hours
    depending on --equilibration-ns and system size.
    """
    import rich
    from rich import pretty

    from asapdiscovery.alchemy.cli.utils import print_header
    from asapdiscovery.alchemy.membrane import build_membrane_system

    pretty.install()
    console = rich.get_console()
    print_header(console)

    spec = _load_spec(preset, spec_file)
    settings = _load_settings(settings_file)
    try:
        build_membrane_system(
            receptor_pdb=receptor,
            spec=spec,
            output_prefix=output_prefix,
            equilibration_ns=equilibration_ns,
            pocket_ligand=pocket_ligand,
            settings=settings,
        )
    except ValueError as error:
        raise click.ClickException(str(error))
    console.print(
        f"\nBuilt {output_prefix}_membrane_system.pdb -- plan against it with:\n"
        f"  asap-alchemy plan --membrane-receptor {output_prefix}_membrane_system.pdb "
        "--ligands <posed.sdf> --name <name>"
    )
