import click
from asapdiscovery.simulation.cli.create import create
from asapdiscovery.simulation.cli.plan import plan
from asapdiscovery.simulation.cli.status import status
from asapdiscovery.simulation.cli.submit import submit


@click.group()
def cli():
    pass


cli.add_command(create)
cli.add_command(plan)
cli.add_command(submit)
# cli.add_command(results)
cli.add_command(status)

if __name__ == "__main__":
    cli()
