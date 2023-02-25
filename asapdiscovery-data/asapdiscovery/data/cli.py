import click


@click.group()
def cli(help="Command-line interface for asapdiscovery-data"):
    ...


from .aws.cli import aws

# import subcommands to register them
from .postera.cli import postera
