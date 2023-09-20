from typing import Optional

from alchemiscale import Scope, ScopedKey
from openmm.app import ForceField, Modeller, PDBFile

from .schema.fec import (
    AlchemiscaleFailure,
    AlchemiscaleResults,
    AlchemiscaleSettings,
    FreeEnergyCalculationNetwork,
    TransformationResult,
)
from .schema.forcefield import ForceFieldParams


def create_protein_only_system(input_pdb_path: str, ff_params: ForceFieldParams):
    # Input Files
    pdb = PDBFile(input_pdb_path)
    forcefield = ForceField(*ff_params.ff_xmls)

    # Prepare the Simulation
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.addSolvent(forcefield, padding=ff_params.padding, model=ff_params.model)
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=ff_params.nonbondedMethod,
        nonbondedCutoff=ff_params.nonbondedCutoff,
        constraints=ff_params.constraints,
        rigidWater=ff_params.rigidWater,
        ewaldErrorTolerance=ff_params.ewaldErrorTolerance,
        hydrogenMass=ff_params.hydrogenMass,
    )
    return system


class AlchemiscaleHelper:
    """
    A convenience class to handle alchemiscale submissions restarts and results gathering.
    """

    def __init__(self, api_url: str = "https://api.alchemiscale.org"):
        """
        Create the client which will be used for the rest of the queries
        """
        from alchemiscale import AlchemiscaleClient

        # load the settings from the environment
        settings = AlchemiscaleSettings()
        # connect to the client
        self._client = AlchemiscaleClient(
            api_url=api_url,
            identifier=settings.ALCHEMISCALE_ID,
            key=settings.ALCHEMISCALE_KEY,
        )

    def create_network(
        self, planned_network: FreeEnergyCalculationNetwork, scope: Scope
    ) -> FreeEnergyCalculationNetwork:
        """
        Create the given network on the alchemiscale instance no tasks are actioned yet.

        Args:
            planned_network: The network to submit to alchemiscale.
            scope: The alchemiscale scope the calculation to be submitted to.

        Returns
            A copy of the network submit with the results object updated.
        """
        # store a copy of the input data so we can add the results holder
        network_data = planned_network.dict()

        # build the network which we can submit
        fec_network = planned_network.to_alchemical_network()
        # build the network on alchemiscale and get a key
        network_key = self._client.create_network(fec_network, scope)

        # save the scoped key to find the network later and build the planned network again
        alchem_result = AlchemiscaleResults(network_key=network_key)
        network_data["results"] = alchem_result
        # work around the frozen object
        result_network = FreeEnergyCalculationNetwork(**network_data)
        return result_network

    def action_network(
        self, planned_network: FreeEnergyCalculationNetwork
    ) -> list[Optional[ScopedKey]]:
        """
        For the given network which is already stored on alchemiscale create and action tasks.

        Args:
            planned_network: The network which should action tasks for.

        Returns:
            A list of actioned tasks for this network.

        """
        network_key = planned_network.results.network_key

        tasks = []
        for tf_sk in self._client.get_network_transformations(network_key):
            # the factory defines how many times the task should be repeated, so total runs is 1 + no of repeats
            tasks.extend(
                self._client.create_tasks(tf_sk, count=planned_network.n_repeats + 1)
            )

        # now action the tasks to ensure they are picked up by compute.
        actioned_tasks = self._client.action_tasks(tasks, network_key)
        return actioned_tasks

    def network_status(
        self, planned_network: FreeEnergyCalculationNetwork
    ) -> dict[str, int]:
        """
        Get the status of the network from alchemiscale.

        Args:
            planned_network: The network which we should look up in alchemiscale.

        Returns:
            A dict of the status type and the number of instances.
        """
        network_key = planned_network.results.network_key
        return self._client.get_network_status(network_key)

    def collect_results(
        self, planned_network: FreeEnergyCalculationNetwork
    ) -> FreeEnergyCalculationNetwork:
        """
        Collect the results for the given network.

        Args:
            planned_network: The network whose results we should collect.

        Returns:
            A FreeEnergyCalculationNetwork with all current results. If any are missing and allow missing is false an error is raised.
        """
        # collect results following the notebook from openFE
        results = []
        for tf_sk in self._client.get_network_transformations(
            planned_network.results.network_key
        ):
            raw_result = self._client.get_transformation_results(tf_sk)
            if raw_result is None:
                continue
            # format into our custom result schema and save
            estimate = raw_result.get_estimate()
            uncertainty = raw_result.get_uncertainty()
            # work out the name of the molecules and the phase of the calculation
            individual_runs = list(raw_result.data.values())
            # track the phase to correctly work out the total relative energy as complex - solvent
            if "protein" in individual_runs[0][0].inputs["stateA"].components:
                phase = "complex"
            else:
                phase = "solvent"

            # extract the names of the end state ligands to build the affinity estimate graph
            name_a = individual_runs[0][0].inputs["stateA"].components["ligand"].name
            name_b = individual_runs[0][0].inputs["stateB"].components["ligand"].name

            results.append(
                TransformationResult(
                    ligand_a=name_a,
                    ligand_b=name_b,
                    phase=phase,
                    estimate=estimate,
                    uncertainty=uncertainty,
                )
            )

        # save to a new results object as they are frozen
        alchem_results = AlchemiscaleResults(
            network_key=planned_network.results.network_key, results=results
        )
        network_with_results = FreeEnergyCalculationNetwork(
            **planned_network.dict(exclude={"results"}), results=alchem_results
        )

        return network_with_results

    def collect_errors(
        self,
        planned_network: FreeEnergyCalculationNetwork,
    ) -> list[AlchemiscaleFailure]:
        """
        Collect errors and tracebacks from failed tasks.

        Args:
            planned_network: Network to get failed tasks from.

        Returns:
            List of AlchemiscaleFailure objects with errors and tracebacks for the failed tasks in the network.
        """
        network_key = planned_network.results.network_key
        errored_tasks = self._client.get_network_tasks(network_key, status="error")

        error_data = list()
        for task in errored_tasks:
            for err_result in self._client.get_task_failures(task):
                for failure in err_result.protocol_unit_failures:
                    failure = AlchemiscaleFailure(
                        network_key=network_key,
                        task_key=task.gufe_key,
                        unit_key=failure.source_key,
                        dag_result_key=err_result.key,
                        error=failure.exception,
                        traceback=failure.traceback,
                    )
                    error_data.append(failure)
        return error_data
