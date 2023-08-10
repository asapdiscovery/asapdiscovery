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
@click.option(
    "--allow-missing/--no-allow-missing",
    "allow_missing",
    default=False,
    help="If we should allow missing results when gathering from alchemiscale.",
)
def gather(network: str, allow_missing: bool):
    """
    Gather the results from alchemiscale for the given network.

    Note: An error is raised if all calculations have not finished and allow-missing is False.

    Args:
        network: The of the JSON file containing the FreeEnergyCalculationNetwork whos results we should gather.
        allow_missing: If we should allow missing results when trying to gather the network.

    Raises:
        Runtime error if all calculations are not complete and allow missing is False.
    """
    from asapdiscovery.simulation.schema.fec import FreeEnergyCalculationNetwork
    from asapdiscovery.simulation.utils import AlchemiscaleHelper

    # launch the helper which will try to login
    click.echo("Connecting to Alchemiscale...")
    client = AlchemiscaleHelper()

    # load the network
    planned_network = FreeEnergyCalculationNetwork.from_file(network)

    # show the network status
    status = client.network_status(planned_network=planned_network)
    if not allow_missing and len(status) > 1:
        raise RuntimeError(
            "Not all calculations have finished, to collect the current results use the flag `--allow_missing`."
        )

    click.echo(
        f"Gathering network results from Alchemiscale instance: {client._client.api_url} with key {planned_network.results.network_key}"
    )
    network_with_results = client.collect_results(planned_network=planned_network)
    click.echo("Results gathered saving to file ...")
    network_with_results.to_file("result_network.json")
