import click


@click.group()
def cli(help="Command-line interface for asapdiscovery"):
    ...


from asapdiscovery.workflows.docking_workflows.cli import (  # noqa: F401, E402, F811
    docking,
)

cli.add_command(docking)

from asapdiscovery.workflows.prep_workflows.cli import (  # noqa: F401, E402, F811
    protein_prep,
)

cli.add_command(protein_prep)

from asapdiscovery.alchemy.cli.cli import alchemy  # noqa: F401, E402, F811

cli.add_command(alchemy)


from asapdiscovery.ml.cli import ml  # noqa: F401, E402, F811

cli.add_command(ml)
