"""Regenerate the test files to keep inline with the new schema only consistent data like the network, ligands and receptor and results are
transferred all openfe/asap-alchemy schema data is set to the new model defaults.

You will need local copies of tyk2_small_network.json and tyk2_result_network.json
"""

import json

from asapdiscovery.alchemy.schema.fec import (
    AlchemiscaleResults,
    FreeEnergyCalculationNetwork,
    TransformationResult,
)
from openff.units import unit


def main():

    # convert the normal network keeping only the network, receptor and dataset_name
    old_schema = json.load(open("tyk2_small_network.json"))
    new_schema = FreeEnergyCalculationNetwork(
        network=old_schema["network"],
        receptor=old_schema["receptor"],
        dataset_name=old_schema["dataset_name"],
    )
    new_schema.to_file("NEW_tyk2_small_network.json")

    # now do the same again but we need to be careful with the results which have units
    old_schema = json.load(open("tyk2_result_network.json"))
    results = []
    for result in old_schema["results"]["results"]:
        new_result = TransformationResult(
            ligand_a=result["ligand_a"],
            ligand_b=result["ligand_b"],
            phase=result["phase"],
            estimate=result["estimate"]["magnitude"]
            * getattr(unit, result["estimate"]["unit"]),
            uncertainty=result["uncertainty"]["magnitude"]
            * getattr(unit, result["uncertainty"]["unit"]),
        )
        results.append(new_result)
    all_results = AlchemiscaleResults(
        network_key=old_schema["results"]["network_key"], results=results
    )
    new_schema = FreeEnergyCalculationNetwork(
        network=old_schema["network"],
        receptor=old_schema["receptor"],
        dataset_name=old_schema["dataset_name"],
        results=all_results,
    )
    new_schema.to_file("NEW_tyk2_result_network.json")


if __name__ == "__main__":
    main()
