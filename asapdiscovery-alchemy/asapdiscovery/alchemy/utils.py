from typing import TYPE_CHECKING, Optional

import numpy as np
from alchemiscale import Scope, ScopedKey
from asapdiscovery.alchemy.interfaces import AlchemiscaleSettings
from asapdiscovery.alchemy.schema.fec import (
    AlchemiscaleFailure,
    AlchemiscaleResults,
    FreeEnergyCalculationNetwork,
    TransformationResult,
)
from asapdiscovery.alchemy.schema.forcefield import ForceFieldParams
from openmm.app import ForceField, Modeller, PDBFile

if TYPE_CHECKING:
    from asapdiscovery.data.schema.complex import PreppedComplex
    from asapdiscovery.data.schema.ligand import Ligand
    from asapdiscovery.data.schema.target import PreppedTarget
    from openff.bespokefit.workflows import BespokeWorkflowFactory


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

    def __init__(
        self, identifier: str, key: str, api_url: str = "https://api.alchemiscale.org"
    ):
        """
        Create the client which will be used for the rest of the queries.

        Args:
            identifier: Your personal alchemiscale ID used to login.
            key: Your personal alchemiscale KEY used to login.
            api_url: The URL of the alchemiscale instance to connect to.
        """
        from alchemiscale import AlchemiscaleClient

        # connect to the client
        self._client = AlchemiscaleClient(
            api_url=api_url,
            identifier=identifier,
            key=key,
        )

    @classmethod
    def from_settings(cls, settings: Optional[AlchemiscaleSettings] = None):
        if settings is None:
            settings = AlchemiscaleSettings()
        return cls(
            api_url=settings.ALCHEMISCALE_URL,
            key=settings.ALCHEMISCALE_KEY,
            identifier=settings.ALCHEMISCALE_ID,
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

    def get_actioned_weights(self) -> list[float]:
        """
        For all waiting/running networks, return the weights associated with them.

        Returns:
            A list of floats that are associated with all currently waiting/running networks.
        """
        finished_counter = 0
        active_network_weights = []

        # get all networks that we're able to see
        for key in self._client.query_networks():
            network_stats = self._client.get_network_status(
                network=key, visualize=False
            )
            n_running = network_stats.get("running", 0)
            n_waiting = network_stats.get("waiting", 0)

            # if we've encountered 5 stale networks we've probably reached the finished ones
            # can safely stop collecting weights
            if n_running == 0 and n_waiting == 0:
                finished_counter += 1
            if finished_counter == 5:
                break

            active_network_weights.append(self._client.get_network_weight(network=key))

        return active_network_weights

    def action_network(
        self, planned_network: FreeEnergyCalculationNetwork, repeats: int
    ) -> list[Optional[ScopedKey]]:
        """
        For the given network which is already stored on alchemiscale create and action tasks.

        Args:
            planned_network: The network which should action tasks for.
            repeats: The number of times each transformation should be run, results will be averaged over the n
                successful repeats.

        Returns:
            A list of actioned tasks for this network.
        """
        network_key = planned_network.results.network_key

        tasks = []
        for tf_sk in self._client.get_network_transformations(network_key):
            tasks.extend(self._client.create_tasks(tf_sk, count=repeats))

        # now action the tasks to ensure they are picked up by compute.
        actioned_tasks = self._client.action_tasks(tasks, network_key)
        return actioned_tasks

    def network_status(
        self,
        network_key: str,
    ) -> dict[str, int]:
        """
        Get the status of the network from alchemiscale.

        Args:
            network_key: The network key belonging to the network which we should look up in alchemiscale.

        Returns:
            A dict of the status type and the number of instances.
        """
        return self._client.get_network_status(network_key)

    def network_exists(self, network_key: str) -> bool:
        """
        Check if a network exists on alchemiscale.

        Args:
            network_key: The network key belonging to the network which we should look up in alchemiscale.

        Returns:
            True if the network exists, False otherwise.
        """
        return self._client.check_exists(network_key)

    def restart_tasks(
        self,
        planned_network: FreeEnergyCalculationNetwork,
        tasks: Optional[list[ScopedKey]] = None,
    ) -> list[ScopedKey]:
        """
        Restart errored tasks on alchemiscale for this network.

        Args:
            planned_network: The network which we should look up in alchemiscale.
            tasks: ScopedKeys to limit restarts to; if empty, then all errored tasks restarted.

        Returns:
            list of ScopedKeys for the tasks that were restarted
        """
        network_key = planned_network.results.network_key
        errored_tasks = self._client.get_network_tasks(network_key, status="error")

        if tasks:
            to_restart = list(set(tasks) & set(errored_tasks))
        else:
            to_restart = errored_tasks

        restarted_tasks = self._client.set_tasks_status(to_restart, status="waiting")

        return restarted_tasks

    def collect_results(
        self,
        planned_network: Optional[FreeEnergyCalculationNetwork] = None,
        network_key: Optional[str] = None,
    ) -> FreeEnergyCalculationNetwork:
        """
        Collect the results for the given network.

        Args:
            planned_network: The network whose results we should collect.
            network_key: The `alchemsicale` network key for the network whose results we should collect.

        Returns:
            A FreeEnergyCalculationNetwork with all current results. If any are missing and allow missing is false an error is raised.
        """
        # collect results following the notebook from openFE
        results = []

        if network_key:
            raise NotImplementedError(
                "ASAP-Alchemy gather using network keys (-nk) is currently not implemented."
            )

        if planned_network and network_key:
            raise ValueError("Provide only one of `planned_network` or `network_key`")
        if not network_key and not planned_network:
            raise ValueError(
                "Need to define one of `planned_network` or `network_key`."
            )

        if planned_network:
            network_key = planned_network.results.network_key

        alchemiscale_network_results = self._client.get_network_results(
            network_key
        ).items()
        # use the process pool api point to gather all transformations for the network
        for _, raw_result in alchemiscale_network_results:
            if raw_result is None:
                continue
            # format into our custom result schema and save
            estimate = raw_result.get_estimate()
            uncertainty = raw_result.get_uncertainty()
            # if there is a single repeat the error is 0.0 so extract the mbar error
            if uncertainty.m == 0.0:
                uncertainty = [
                    edge[0].outputs["unit_estimate_error"]
                    for edge in raw_result.data.values()
                ][0]

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
            # print(individual_runs[0][0].inputs["stateB"].components["ligand"], name_b)

            # if end state ligands did not have names, use SMILES instead
            if not name_a:
                name_a = (
                    individual_runs[0][0].inputs["stateA"].components["ligand"].smiles
                )
            if not name_b:
                name_b = (
                    individual_runs[0][0].inputs["stateB"].components["ligand"].smiles
                )

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
        alchem_results = AlchemiscaleResults(network_key=network_key, results=results)

        if planned_network:
            network_with_results = FreeEnergyCalculationNetwork(
                **planned_network.dict(exclude={"results"}), results=alchem_results
            )

        return network_with_results

    def collect_errors(
        self,
        network_key: str,
    ) -> list[AlchemiscaleFailure]:
        """
        Collect errors and tracebacks from failed tasks.

        Args:
            planned_network: Network to get failed tasks from.

        Returns:
            List of AlchemiscaleFailure objects with errors and tracebacks for the failed tasks in the network.
        """
        errored_tasks = self._client.get_network_tasks(network_key, status="error")

        error_data = []
        for task in errored_tasks:
            for err_result in self._client.get_task_failures(task):
                for protocol_failure in err_result.protocol_unit_failures:
                    failure = AlchemiscaleFailure(
                        network_key=network_key,
                        task_key=task,
                        unit_key=protocol_failure.source_key,
                        dag_result_key=err_result.key,
                        error=protocol_failure.exception,
                        traceback=protocol_failure.traceback,
                    )
                    error_data.append(failure)
        return error_data

    def cancel_actioned_tasks(
        self, network_key: ScopedKey, hard: bool = False
    ) -> list[ScopedKey]:
        """
        Cancel all currently actioned tasks on a network to stop all future compute.

        Notes:
            This removes the networks from the view of `asap-alchemy status -a`.
            To run these tasks again they must be actioned.

        Args:
            network_key: The alchemiscale network key who's actioned tasks should be canceled.
            hard: If true, all running/waiting tasks will be deleted. Warning: these are not retrievable/re-runnable.

        Returns:
            A list of the ScopedKeys of all canceled tasks.
        """
        if hard:
            running_tasks = self._client.get_network_tasks(
                network=network_key, status="running"
            )
            waiting_tasks = self._client.get_network_tasks(
                network=network_key, status="waiting"
            )
            tasks_to_delete = running_tasks + waiting_tasks
            if tasks_to_delete:
                canceled_tasks = self._client.set_tasks_status(
                    tasks=running_tasks, status="deleted"
                )
            else:
                canceled_tasks = []
        else:
            actioned_tasks = self._client.get_network_actioned_tasks(
                network=network_key
            )
            if actioned_tasks:
                canceled_tasks = self._client.cancel_tasks(
                    tasks=actioned_tasks, network=network_key
                )
            else:
                canceled_tasks = []

        return canceled_tasks

    def adjust_weight(
        self, network_key: ScopedKey, weight: float
    ) -> tuple[float, float]:
        """
        Adjust the weight of a network to influence how often its tasks get actioned
        by the alchemiscale scheduler.

        Args:
            network_key: The alchemiscale network key that should have its weight adjusted.
            weight: The weight (a float between 0.0 and 1.0) that should be assigned.

        Returns:
            The new weight that is assigned to the network.
            The weight that was previously assigned to the network.
        """
        old_weight = self._client.get_network_weight(network=network_key)

        self._client.set_network_weight(network=network_key, weight=weight)

        return self._client.get_network_weight(network=network_key), old_weight


def select_reference_for_compounds(
    ligands: list["Ligand"],
    references: list["PreppedComplex"],
    check_openmm: bool = True,
) -> tuple["PreppedComplex", "Ligand"]:
    """
    From a collection of ligands and a list of `Complex`es, return the `Complex` that is the most similar to
    the largest of the query ligands and should be used to constrain the generated poses.

    Args:
        ligands: The list of ligands in the alchemy network we need a reference for.
        references: The list of prepped references we can select from.
        check_openmm: If we should try and build an openMM system for the complex to ensure if can be simulated.

    Returns:
        The PreppedComplex most suitable for the input ligands and the largest ligand that it was selected to match.
    """
    from asapdiscovery.data.operators.selectors.mcs_selector import sort_by_mcs

    # sort the ligands by the number of atoms
    compounds_by_size = [
        (ligand.to_rdkit().GetNumAtoms(), ligand) for ligand in ligands
    ]
    compounds_by_size.sort(key=lambda x: x[0], reverse=True)

    # find that largest ligand's closest reference
    ref_ligands = [ref.ligand for ref in references]
    sorted_index = sort_by_mcs(
        reference_ligand=compounds_by_size[0][1], target_ligands=ref_ligands
    )

    sorted_complexs = np.asarray(references)[sorted_index]
    best_complex = sorted_complexs[0]
    if check_openmm:
        # check each complex can make a valid system in openmm and return the first success
        for ref_complex in sorted_complexs:
            if is_valid_receptor_system(target=ref_complex.target):
                best_complex = ref_complex
                break

    return best_complex, compounds_by_size[0][1]


def get_similarity(ligand_a: "Ligand", ligand_b: "Ligand") -> float:
    """
    Get the ECFP6 tanimoto similarity between two ligands.
    """
    from rdkit import DataStructs
    from rdkit.Chem import AllChem

    radius = 3  # ECFP6 because of diameter instead of radius
    simi = DataStructs.FingerprintSimilarity(
        AllChem.GetMorganFingerprintAsBitVect(ligand_a.to_rdkit(), radius),
        AllChem.GetMorganFingerprintAsBitVect(ligand_b.to_rdkit(), radius),
    )

    return round(simi, 2)


def is_valid_receptor_system(target: "PreppedTarget") -> bool:
    """
    Check we can build an openMM for the given target.

    Args:
        target: The prepped target for which we should try and build a system.

    Returns:
        `True` if a system can be build without error else `False`
    """
    import tempfile

    from openmm import app

    # load current standard force fields in openFE 17.04.2024
    forcefield = app.ForceField(
        *[
            "amber/ff14SB.xml",  # ff14SB protein force field
            "amber/tip3p_standard.xml",  # TIP3P and recommended monovalent ion parameters
            "amber/tip3p_HFE_multivalent.xml",  # for divalent ions
            "amber/phosaa10.xml",  # Handles THE TPO
        ]
    )
    with tempfile.NamedTemporaryFile(suffix=".pdb") as fp:
        target.to_pdb_file(fp.name)
        pdb = app.PDBFile(fp.name)
        try:
            _ = forcefield.createSystem(pdb.topology)
        except Exception:
            return False

    return True


def extract_custom_ligand_network(csv_file: str) -> list[tuple[str, str]]:
    """
    Extract the named transformations from the csv file which should be used during the network planning stage.

    Args:
        csv_file: The path to the csv file we should extract the named edges from.

    Returns:
        A list of tuples containing the names of the ligands which make up each edge of the network.
    """
    import pandas as pd

    # do some checks on the input data.
    try:
        custom_network_df = pd.read_csv(csv_file, header=None)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Custom network file {csv_file} is empty.")

    edges = []
    for i, row in custom_network_df.iterrows():
        row_data = [str(value).strip() for value in row.values]
        if row.isnull().values.any() or any(entry.rsplit() == [] for entry in row_data):
            raise ValueError(
                f"Custom network file contains an empty entry at index {i+1}"
            )
        # if no errors extract the data
        edges.append(tuple(row_data))
    return edges


class BespokeFitHelper:
    """
    A convenience class to handle BespokeFit submissions restarts and results gathering.
    """

    def __init__(self):
        from openff.bespokefit.executor.client import BespokeFitClient, Settings

        self._client = BespokeFitClient(settings=Settings())

    def submit_ligands(
        self,
        network: FreeEnergyCalculationNetwork,
        bespokefit_protocol: "BespokeWorkflowFactory",
    ) -> FreeEnergyCalculationNetwork:
        """
        Submit the ligands from the network to a running BespokeFit server.

        Notes:
            We assume that relevant BespokeFit environment settings are exposed.

        Args:
            network: The network with ligands which should be submited.

        Returns:
            The network with the bespokefit_ids added as a tag on each ligand.
        """
        from openff.toolkit import Molecule

        for ligand in network.network.ligands:
            # create the job schema
            bespoke_job = bespokefit_protocol.optimization_schema_from_molecule(
                molecule=Molecule.from_rdkit(ligand.to_rdkit()),
                index=ligand.compound_name,
            )
            # submit the job and save the task ID
            response = self._client.submit_optimization(input_schema=bespoke_job)
            ligand.tags["bespokefit_id"] = response

        return network

    def gather_results(
        self, network: FreeEnergyCalculationNetwork
    ) -> FreeEnergyCalculationNetwork:
        """
        Gather the bespoke parameters for the ligands in the network file from a BespokeFit server.

        Args:
            network: The network with ligands who's results we should gather.

        Returns:
            The network with the bespoke parameters stored on the ligands.
        """
        from asapdiscovery.data.schema.identifiers import (
            BespokeParameter,
            BespokeParameters,
        )

        for ligand in network.network.ligands:
            if "bespokefit_id" in ligand.tags:
                bespoke_result = self._client.get_optimization(
                    ligand.tags["bespokefit_id"]
                )
                # we can only save the parameters if the optimisation has finished
                if bespoke_result.status == "success":

                    refit_parameters = bespoke_result.results.refit_parameter_values
                    # make sure we use the force field which we fit to
                    # this was taken from the network originally
                    bespoke_parameters = BespokeParameters(
                        base_force_field=network.forcefield_settings.small_molecule_forcefield
                    )
                    for parameter, values in refit_parameters.items():
                        bespoke_parameter = BespokeParameter(
                            interaction=parameter.type,
                            smirks=parameter.smirks,
                            values={key: value.m for key, value in values.items()},
                            units="kilocalories_per_mole",
                        )
                        bespoke_parameters.parameters.append(bespoke_parameter)
                    # save the parameters back to the ligand
                    ligand.bespoke_parameters = bespoke_parameters
        return network

    def status(self, network: FreeEnergyCalculationNetwork) -> dict[str, int]:
        """
        Gather the status of the parameter optimization for the ligands in the network.

        Args:
            network: The network with ligands that we want the status for.

        Returns
            A dictionary with the count per status type for the ligands in the network.
        """
        states = {"success": 0, "running": 0, "waiting": 0, "errored": 0}
        for ligand in network.network.ligands:
            if "bespokefit_id" in ligand.tags:
                response = self._client.get_optimization(ligand.tags["bespokefit_id"])
                states[response.status] += 1

        return states
