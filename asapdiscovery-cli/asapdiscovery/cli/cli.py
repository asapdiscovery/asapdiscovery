import click


@click.group()
def cli(help="Command-line interface for asapdiscovery"):
    ...


from asapdiscovery.docking.cli import cli # noqa: F401, E402, F811 
