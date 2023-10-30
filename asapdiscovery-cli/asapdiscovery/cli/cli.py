import click


@click.group()
def cli(help="Command-line interface for asapdiscovery"):
    ...


from asapdiscovery.docking.cli import cli
