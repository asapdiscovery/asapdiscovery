import click


@click.group()
def cli(help="Command-line interface for asapdiscovery"):
    ...


from asapdiscovery.docking.cli import docking  # noqa: F401, E402, F811

cli.add_command(docking)

from asapdiscovery.modeling.cli import modeling  # noqa: F401, E402, F811

cli.add_command(modeling)

from asapdiscovery.alchemy.cli.cli import alchemy  # noqa: F401, E402, F811

cli.add_command(alchemy)