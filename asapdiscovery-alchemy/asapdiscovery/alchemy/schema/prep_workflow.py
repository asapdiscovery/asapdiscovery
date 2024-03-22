import copy
from typing import Any, Literal, Optional, Union

import rich
from asapdiscovery.alchemy.schema.base import _SchemaBase
from asapdiscovery.data.operators.state_expanders.protomer_expander import EpikExpander
from asapdiscovery.data.operators.state_expanders.stereo_expander import StereoExpander
from asapdiscovery.data.schema.complex import PreppedComplex
from asapdiscovery.data.schema.ligand import Ligand
from asapdiscovery.docking.schema.pose_generation import (
    OpenEyeConstrainedPoseGenerator,
    PosedLigands,
    RDKitConstrainedPoseGenerator,
)
from pydantic import Field
from rich import pretty
from rich.padding import Padding


class _AlchemyPrepBase(_SchemaBase):
    """
    A base class for the Alchemy prep workflow to capture the settings for the factory and results objects.
    """

    type: Literal["_AlchemyPrepBase"] = "_AlchemyPrepBase"

    stereo_expander: Optional[StereoExpander] = Field(
        StereoExpander(),
        description="A class to expand the stereo"
        "chemistry of the ligands. This stage will be skipped if set to `None`.",
    )
    charge_expander: Optional[EpikExpander] = Field(
        None,
        description="The charge and tautomer expander that"
        "should be applied to the ligands. This stage will be skipped if set to `None`.",
    )
    pose_generator: Union[
        OpenEyeConstrainedPoseGenerator, RDKitConstrainedPoseGenerator
    ] = Field(
        RDKitConstrainedPoseGenerator(),
        description="The method "
        "to generate the initial poses for the molecules for FEC.",
    )
    core_smarts: Optional[str] = Field(
        None,
        description="The SMARTS string which should be used to identify the MCS between the "
        "input and reference ligand if not provided the MCS will be automatically generated. SMARTS strings can be created manually, or with e.g. ChemDraw or https://smarts.plus/.",
    )
    strict_stereo: bool = Field(
        True,
        description="Molecules will have conformers generated if their stereo chemistry matches the input molecule.",
    )
    n_references: int = Field(
        3,
        description="The number of experimental reference molecules we should try to generate "
        "poses for.",
    )


class AlchemyDataSet(_AlchemyPrepBase):
    """
    A dataset of prepared ligands ready for FEC generated by the AlchemyPrepWorkflow.
    """

    type: Literal["AlchemyDataSet"] = "AlchemyDataSet"

    dataset_name: str = Field(..., description="The name of the dataset.")
    reference_complex: PreppedComplex = Field(
        ...,
        description="The prepared complex which was used in pose generation including the crystal reference ligand.",
    )
    input_ligands: list[Ligand] = Field(
        ..., description="The list of ligands input to the workflow."
    )
    posed_ligands: list[Ligand] = Field(
        ..., description="The list of Ligands with their generated poses."
    )
    failed_ligands: Optional[dict[str, list[Ligand]]] = Field(
        None,
        description="A list of ligands removed from the workflow stored by the stage that removed them.",
    )
    provenance: dict[str, dict[str, Any]] = Field(
        ...,
        description="The provenance information for each of the stages in the workflow stored by the stage name.",
    )

    def save_posed_ligands(self, filename: str):
        """
        Save the posed ligands to an SDF file using openeye.

        Parameters
        ----------
        filename: The name of the SDF the ligands should be saved to.
        """
        from asapdiscovery.data.backend.openeye import save_openeye_sdfs

        oemols = [ligand.to_oemol() for ligand in self.posed_ligands]
        save_openeye_sdfs(oemols, filename)


class AlchemyPrepWorkflow(_AlchemyPrepBase):
    """
    A factory to handle the state expansion and constrained pose generation used as inputs to the Alchemy workflow.
    """

    type: Literal["AlchemyPrepWorkflow"] = "AlchemyPrepWorkflow"

    @staticmethod
    def _validate_ligands(ligands: list[Ligand]) -> list[Ligand]:
        """
        For the given set of ligands make sure that the docked ligand is the intended ligand target i.e does the
        3D stereo match what we intend at input.
        """
        failed_ligands = []
        for ligand in ligands:
            # check the original fixed inchikey against the current one
            if ligand.provenance.fixed_inchikey != ligand.fixed_inchikey:
                failed_ligands.append(ligand)
        return failed_ligands

    @staticmethod
    def _sort_similar_molecules(
        reference_ligand: Ligand, experimental_ligands: list[Ligand]
    ) -> Ligand:
        """
        Sort the list of experimental ligands by MCS overlap with the reference crystal ligand to determine the order
        in which the structures should be generated.

        Args:
            reference_ligand: The crystal structure ligand which will be the basis for the constrained pose generation.
            experimental_ligands: The list experimental ligands we would like to add to this dataset.

        Returns:

        """
        import numpy as np
        from asapdiscovery.data.operators.selectors.mcs_selector import sort_by_mcs

        # use the mcs code to get the ordered indices of the matches
        sort_idx = sort_by_mcs(
            reference_ligand=reference_ligand,
            target_ligands=experimental_ligands,
            structure_matching=False,
        )

        ligands_sorted = np.asarray(experimental_ligands)[sort_idx]

        return ligands_sorted

    def pose_experimental_molecules(
        self,
        reference_complex: PreppedComplex,
        experimental_ligands: list[Ligand],
        processors: int = 1,
    ) -> list[Ligand]:
        """
        Iteratively try and generate poses for the experimental ligands until we have `self.n_references` posed.

        Args:
            reference_complex: The complex with the crystal structure which is used to constrain the generated poses.
            experimental_ligands: The list of experimental ligands ordered in list of priority.
            processors: The number of processor available to the pose generator.

        Returns:
            A list of posed experimental ligands.
        """
        posed_refs = []
        # run in batches so we don't try and generate poses for everything but run faster than serial
        batch_size = self.n_references * 2
        for i in range(0, len(experimental_ligands), batch_size):
            ligand_batch = experimental_ligands[i : i + batch_size]
            poses = self.pose_generator.generate_poses(
                prepared_complex=reference_complex,
                ligands=experimental_ligands[i : i + batch_size],
                core_smarts=self.core_smarts,
                processors=processors,
            )

            posed_ligands = poses.posed_ligands

            if self.strict_stereo:
                # remove the stereo issue molecules before checking how many have been posed
                stereo_fails = AlchemyPrepWorkflow._validate_ligands(
                    ligands=posed_ligands
                )
                posed_ligands = AlchemyPrepWorkflow._remove_fails(
                    posed_ligands=posed_ligands, stereo_issue_ligands=stereo_fails
                )

            # skip to the next batch if none were generated
            if not posed_ligands:
                continue

            posed_ligands_by_inchi = {
                ligand.provenance.fixed_inchikey: ligand for ligand in posed_ligands
            }
            # ligands are not in order so check them in the input ordering
            for ligand in ligand_batch:
                try:
                    posed_refs.append(
                        posed_ligands_by_inchi[ligand.provenance.fixed_inchikey]
                    )
                except KeyError:
                    continue

            # stop if we have enough posed ligands
            if len(posed_refs) >= self.n_references:
                break

        # finally return either when we have enough or run out of ligands
        return posed_refs[: self.n_references]

    @staticmethod
    def _remove_fails(
        posed_ligands: list[Ligand], stereo_issue_ligands: list[Ligand]
    ) -> list[Ligand]:
        """
        A helper method to remove ligands from the posed list which are in the stereo issue list.

        Args:
            posed_ligands: A list of posed ligands which should be filtered.
            stereo_issue_ligands: The list of ligands with stereo issues which should be removed from the posed list.

        Returns:
            A list of posed ligands which have correct and consistent stereo.
        """
        # we need to carefully remove the molecules from the posed_ligands list
        failed_hash = [
            ligand.provenance.fixed_inchikey for ligand in stereo_issue_ligands
        ]
        final_ligands = [
            mol
            for mol in posed_ligands
            if mol.provenance.fixed_inchikey not in failed_hash
        ]
        return final_ligands

    @staticmethod
    def _deduplicate_experimental_ligands(
        posed_ligands: list[Ligand], experimental_ligands: list[Ligand]
    ) -> list[Ligand]:
        """
        Remove duplicated ligands in the experimental list which have already been posed.

        Notes:
            This function marks the duplicated ligands in the posed list as experimental which helps with predictions
            later in the workflow.

        Args:
            posed_ligands: A list of posed ligands.
            experimental_ligands: A list of experimental ligands which can be posed.

        Returns:
            The deduplicated list of experimental ligands which should be posed.

        """
        # find the protocol name so we can mark the experimental ligands
        protocol_name = experimental_ligands[0].tags.get("cdd_protocol")
        posed_ligand_by_hash = {
            ligand.provenance.fixed_inchikey: ligand for ligand in posed_ligands
        }
        final_exp_ligands = []
        for ligand in experimental_ligands:
            ligand_hash = ligand.provenance.fixed_inchikey
            if ligand_hash not in posed_ligand_by_hash:
                final_exp_ligands.append(ligand)
            else:
                posed_ligand_by_hash[ligand_hash].tags.update(
                    {"experimental": "True", "cdd_protocol": protocol_name}
                )

        return final_exp_ligands

    def create_alchemy_dataset(
        self,
        dataset_name: str,
        ligands: list[Ligand],
        reference_complex: PreppedComplex,
        processors: int = 1,
        reference_ligands: Optional[list[Ligand]] = None,
    ) -> AlchemyDataSet:
        """
        Run the set of input ligands through the state enumeration and pose generation workflow to create a set of posed
        ligands ready for ASAP-Alchemy.

        Notes:
            Ligands with experimental data can be supplied via `reference_ligands`, poses will be generated
            until `self.n_references` have been successfully added. The ligands will be sorted by their MCS overlap with
            the crystal reference ligand to ensure a pose can be generated.

        Args:
            dataset_name: The name which should be given to this dataset.
            ligands: The list of input ligands which should be run through the workflow.
            reference_complex: The prepared target crystal structure with a reference ligand which the poses should be
                constrained to.
            processors: The number of parallel processors that should be used to run the workflow.
            reference_ligands: The list of reference ligands with experimental data which we should also generate
                poses for if `self.n_references` > 0.

        Returns:
            A prepared AlchemyDataset with state expanded ligands posed in the receptor ready for FEC, along with the
            provenance information of the workflow.
        """
        # use rich to display progress
        pretty.install()
        console = rich.get_console()

        # deduplicate ligands first important for FEC networks?
        input_ligands = copy.deepcopy(ligands)
        provenance = {}
        failed_ligands = {}

        # Build the workflow we want to run
        workflow = [
            stage
            for stage in ["stereo_expander", "charge_expander"]
            if getattr(self, stage) is not None
        ]
        # loop over each expansion stage and run
        for stage in workflow:
            expansion_engine = getattr(self, stage)
            stage_status = console.status(
                f"Running state expansion using {expansion_engine.expander_type}"
            )
            stage_status.start()
            ligands = expansion_engine.expand(ligands)
            # log the software versions used
            provenance[expansion_engine.expander_type] = expansion_engine.provenance()
            stage_status.stop()
            console.print(
                f"[[green]✓[/green]] {expansion_engine.expander_type} successful,  number of unique ligands {len(ligands)}."
            )
            console.line()

        # now run the pose generation stage
        pose_status = console.status(
            f"Generating constrained poses using {self.pose_generator.type} for {len(ligands)} ligands."
        )
        # check for stereo in the reference ligand
        if reference_complex.ligand.has_perceived_stereo:
            console.print(
                "[yellow]! WARNING the reference structure is chiral, check output structures carefully! [/yellow]"
            )
            console.line()

        pose_status.start()
        pose_result = self.pose_generator.generate_poses(
            prepared_complex=reference_complex,
            ligands=ligands,
            core_smarts=self.core_smarts,
            processors=processors,
        )
        posed_ligands = pose_result.posed_ligands
        provenance[self.pose_generator.type] = self.pose_generator.provenance()
        # save any failed ligands
        if pose_result.failed_ligands:
            failed_ligands[self.pose_generator.type] = pose_result.failed_ligands
        pose_status.stop()
        console.print(
            f"[[green]✓[/green]] Pose generation successful for {len(pose_result.posed_ligands)}/{len(ligands)}."
        )
        console.line()

        if self.strict_stereo:
            stereo_status = console.status(
                "Removing molecules with inconsistent stereochemistry."
            )
            stereo_status.start()
            stereo_fails = AlchemyPrepWorkflow._validate_ligands(ligands=posed_ligands)
            if stereo_fails:
                # add the new fails to the rest
                failed_ligands["InconsistentStereo"] = stereo_fails
                posed_ligands = AlchemyPrepWorkflow._remove_fails(
                    posed_ligands=posed_ligands, stereo_issue_ligands=stereo_fails
                )

            stereo_status.stop()
            console.print(
                f"[[green]✓[/green]] Stereochemistry filtering complete {len(stereo_fails)} molecules removed."
            )
            console.line()

        if reference_ligands is not None and self.n_references > 0:
            # we need to check if any of the ligands we have already generated poses for are in the experimental list
            # if so mark them with the correct tags for later and remove them from this list
            filter_status = console.status("Removing duplicated reference ligands")
            filter_status.start()
            reference_ligands = AlchemyPrepWorkflow._deduplicate_experimental_ligands(
                posed_ligands=posed_ligands, experimental_ligands=reference_ligands
            )
            filter_status.stop()
            if not reference_ligands:
                console.print("All experimental ligands removed!")

            sort_status = console.status("Sorting reference ligands by MCS overlap.")
            sort_status.start()
            sorted_exp_ligands = AlchemyPrepWorkflow._sort_similar_molecules(
                reference_ligand=reference_complex.ligand,
                experimental_ligands=reference_ligands,
            )
            sort_status.stop()

            pose_status = console.status(
                f"Generating constrained poses using {self.pose_generator.type} for {self.n_references} reference"
                "  ligands."
            )
            pose_status.start()
            # use the wrapper to keep generating poses until we have the correct number
            posed_refs = self.pose_experimental_molecules(
                reference_complex=reference_complex,
                experimental_ligands=sorted_exp_ligands,
                processors=processors,
            )
            pose_status.stop()
            console.print(
                f"[[green]✓[/green]] Pose generation successful for {len(posed_refs)}/{self.n_references} experimental "
                "ligands."
            )
            posed_ligands.extend(posed_refs)

        message = Padding(
            f"Poses successfully generated for {len(posed_ligands)} ligands.",
            (1, 0, 1, 0),
        )
        console.print(message)

        # gather the results
        return AlchemyDataSet(
            **self.dict(exclude={"type"}),
            dataset_name=dataset_name,
            reference_complex=reference_complex,
            input_ligands=input_ligands,
            posed_ligands=posed_ligands,
            failed_ligands=failed_ligands if failed_ligands else None,
            provenance=provenance,
        )
