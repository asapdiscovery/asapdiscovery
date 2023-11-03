import click

from asapdiscovery.alchemy.cli.alchemy import alchemy
from asapdiscovery.alchemy.cli.prep import prep


@click.group()
def cli():
    """The root group for all CLI commands in ASAP-Alchemy"""


cli.add_command(alchemy)
cli.add_command(prep)
