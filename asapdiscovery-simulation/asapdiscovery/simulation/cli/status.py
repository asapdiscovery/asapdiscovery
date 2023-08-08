import click


@click.command()
@click.option(
    "-n",
    "--network",
    type=click.Path(resolve_path=True, readable=True, file_okay=True, dir_okay=False),
    help="The name of the JSON file containing a planned FEC network.",
    default="planned_network.json",
    show_default=True,
)
def status(network: str):
    """
    Get the status of the submitted network on alchemiscale.

    Args:
        network: The name of the JSON file containing the FreeEnergyCalculationNetwork we should check the status of.

    """
    from asapdiscovery.simulation.schema.fec import FreeEnergyCalculationNetwork
    from asapdiscovery.simulation.utils import AlchemiscaleHelper

    # launch the helper which will try to login
    client = AlchemiscaleHelper()
    # load the network
    planned_network = FreeEnergyCalculationNetwork.from_file(network)
    # check the status
    client.network_status(planned_network=planned_network)
