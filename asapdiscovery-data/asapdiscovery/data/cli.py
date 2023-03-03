import click


@click.group()
def cli(help="Command-line interface for asapdiscovery-data"):
    ...


# import subcommands to register them with the cli
from .aws.cli import aws  # noqa: F401, E402
from .postera.cli import postera  # noqa: F401, E402
