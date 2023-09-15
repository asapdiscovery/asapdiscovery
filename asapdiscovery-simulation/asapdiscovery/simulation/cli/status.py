import click


def validate_traceback_flag(ctx, param, value):
    """Validate traceback flag --with-traceback is only used in conjunction with --errors flag."""
    if not ctx.params.get("errors"):
        raise click.UsageError("--with-traceback requires --errors flag to be set.")
    return value


@click.command()
@click.option(
    "-n",
    "--network",
    type=click.Path(resolve_path=True, readable=True, file_okay=True, dir_okay=False),
    help="The name of the JSON file containing a planned FEC network.",
    default="planned_network.json",
    show_default=True,
)
@click.option(
    "--errors",
    is_flag=True,
    default=False,
    help="Output errors from the network, if any.",
)
@click.option(
    "--with-traceback",
    is_flag=True,
    default=False,
    help="Output the tracebacks from the failing tasks. Only usable in conjunction with --errors.",
    callback=validate_traceback_flag,
    is_eager=True,
)
def status(network: str, errors: bool, with_traceback: bool):
    """
    Get the status of the submitted network on alchemiscale.

    Args:
        network: The name of the JSON file containing the FreeEnergyCalculationNetwork we should check the status of.
        errors: Flag to show errors from the tasks.
        with_traceback: Flag to show the complete traceback for the errored tasks.

    """
    from asapdiscovery.simulation.schema.fec import FreeEnergyCalculationNetwork
    from asapdiscovery.simulation.utils import AlchemiscaleHelper

    # launch the helper which will try to login
    client = AlchemiscaleHelper()
    # load the network
    planned_network = FreeEnergyCalculationNetwork.from_file(network)
    # check the status
    client.network_status(planned_network=planned_network)
    # Output errors
    if errors:
        task_errors = client.collect_errors(
            planned_network, with_traceback=with_traceback
        )
        print(task_errors)
