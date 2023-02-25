import click


@click.group()
def cli(help="Command-line interface for asapdiscovery-data"):
    ...

# import subcommands to register them
from .postera.cli import postera
from .aws.cli import aws
