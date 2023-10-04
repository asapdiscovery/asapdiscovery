from typing import Optional

import click


@click.group()
def cli():
    ...

@cli.command(
    name="dock-large"