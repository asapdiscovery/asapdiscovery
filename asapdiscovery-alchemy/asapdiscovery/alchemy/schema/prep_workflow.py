import copy
from typing import Any, Literal, Optional

import rich
from pydantic import Field
from rich import pretty

from asapdiscovery.alchemy.schema.base import _SchemaBase
from asapdiscovery.data.schema_v2.complex import PreppedComplex
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.state_expanders.protomer_expander import EpikExpander
from asapdiscovery.data.state_expanders.stereo_expander import StereoExpander
from asapdiscovery.docking.schema.pose_generation import OpenEyeConstrainedPoseGenerator


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
    pose_generator: OpenEyeConstrainedPoseGenerator = Field(
        OpenEyeConstrainedPoseGenerator(),
        description="The method "
        "to generate the intial poses for the molecules for FEC.",
    )
    core_smarts: Optional[str] = Field(
        None,
        description="The SMARTS string which should be used to identify the MCS between the "
        "input and reference ligand if not provided the MCS will be automatically generated.",
    )
    strict_stereo: bool = Field(
        True,
        description="Molecules will have conformers generated if there stereo chemistry matches the input molecule.",
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
        from asapdiscovery.data.openeye import save_openeye_sdfs

        oemols = [ligand.to_oemol() for ligand in self.posed_ligands]
        save_openeye_sdfs(oemols, filename)


class AlchemyPrepWorkflow(_AlchemyPrepBase):
    """
    A factory to handle the state expansion and constrained pose generation used as inputs to the Alchemy workflow.
    """

    type: Literal["AlchemyPrepWorkflow"] = "AlchemyPrepWorkflow"

    def _validate_ligands(self, ligands: list[Ligand]) -> list[Ligand]:
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

    def create_alchemy_dataset(
        self,
        dataset_name: str,
        ligands: list[Ligand],
        reference_complex: PreppedComplex,
    ) -> AlchemyDataSet:
        """
        Run the set of input ligands through the state enumeration and pose generation workflow to create a set of posed
        ligands ready for ASAP-Alchemy.

        Parameters
        ----------
        dataset_name: The name which should be given to this dataset
        ligands: The list of input ligands which should be run through the workflow
        reference_complex: The prepared target crystal structure with a reference ligand which the poses should be constrained to.

        Returns
        -------
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
        if reference_complex.ligand.has_stereo():
            console.print(
                "[yellow]! WARNING the reference structure is chiral, check output structures carefully! [/yellow]"
            )

        pose_status.start()
        pose_result = self.pose_generator.generate_poses(
            prepared_complex=reference_complex,
            ligands=ligands,
            core_smarts=self.core_smarts,
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
            stereo_fails = self._validate_ligands(ligands=posed_ligands)
            # add the new fails to the rest
            failed_ligands["InconsistentStereo"] = stereo_fails
            # we need to carefully remove the molecules from the posed_ligands list
            failed_hash = [ligand.provenance.fixed_inchikey for ligand in stereo_fails]
            posed_ligands = [
                mol
                for mol in posed_ligands
                if mol.provenance.fixed_inchikey not in failed_hash
            ]
            stereo_status.stop()
            console.print(
                f"[[green]✓[/green]] Stereochemistry filtering complete {len(failed_hash)} molecules removed."
            )
            console.line()

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
