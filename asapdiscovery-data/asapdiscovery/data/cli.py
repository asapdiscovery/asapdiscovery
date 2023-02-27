import click


@click.group()
def cli(help="Command-line interface for asapdiscovery-data"):
    ...


# import subcommands to register them with the cli
from .aws.cli import aws
from .postera.cli import postera
