import abc
from typing import Any, Literal, Optional

from asapdiscovery.data.schema_v2.ligand import Ligand, ReferenceLigand
from asapdiscovery.data.schema_v2.target import PreppedTarget
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt


class _BasicConstrainedPoseGenerator(BaseModel, abc.ABC):
    """An abstract class for other constrained pose generation methods to follow from."""

    type: Literal["_BasicConstrainedPoseGenerator"] = "_BasicConstrainedPoseGenerator"

    class Config:
        allow_mutation = True
        arbitrary_types_allowed = True

    @abc.abstractmethod
    def provenance(self) -> dict[str, Any]:
        """Return the provenance for this pose generation method."""
        ...

    def _generate_poses(
        self,
        receptor: PreppedTarget,
        reference_ligand: ReferenceLigand,
        ligands: list[Ligand],
        core_smarts: str,
    ) -> list[Ligand]:
        """The main worker method which should generate ligand poses in the receptor using the reference ligand where required."""
        ...

    def _validate_ligands(self, ligands: list[Ligand]):
        """
        For the given set of ligands make sure that the docked ligand is the intended ligand target i.e does the
        3D stereo match what we intend at input.
        """
        pass

    def generate_poses(
        self,
        receptor: PreppedTarget,
        reference_ligand: ReferenceLigand,
        ligands: list[Ligand],
        core_smarts: str,
    ) -> list[Ligand]:
        """
        Generate poses for the given list of molecules in the target receptor.

        Note:
            We assume all stereo and states have been expanded and checked by this point.

        Parameters
        ----------
        receptor: The prepared receptor file with the reference ligand removed.
        reference_ligand: The reference ligand which should be used to constrain the pose generation.
        ligands: The list of ligands which require poses in the target receptor.
        core_smarts: The smarts string which should be used to identify the MCS between the ligand and the reference.

        Returns
        -------
            A list of ligands with new poses generated
            # TODO maybe we need some pose generation result object? or to store the results on the ligand schema?
        """
        ligands = self._generate_poses(
            receptor=receptor,
            reference_ligand=reference_ligand,
            ligands=ligands,
            core_smarts=core_smarts,
        )
        self._validate_ligands(ligands=ligands)
        return ligands


class OpenEyeConstrainedPoseGenerator(_BasicConstrainedPoseGenerator):
    type: Literal["OpenEyeConstrainedPoseGenerator"] = "OpenEyeConstrainedPoseGenerator"
    max_confs: PositiveInt = Field(
        1000, description="The maximum number of conformers to try and generate."
    )
    energy_window: PositiveFloat = Field(
        20,
        description="Sets the maximum allowable energy difference between the lowest and the highest energy conformers, in units of kcal/mol",
    )
    clash_cutoff: PositiveFloat = Field(2.0, description="The ")

    def provenance(self) -> dict[str, Any]:
        from openeye import oechem, oeomega

        return {
            "oechem": oechem.OEChemGetVersion(),
            "oeomega": oeomega.OEOmegaGetVersion(),
        }

    def _generate_poses(
        self,
        receptor: PreppedTarget,
        reference_ligand: ReferenceLigand,
        ligands: list[Ligand],
        core_smarts: str,
    ) -> list[Ligand]:
        """Use openeye oeomega to generate constrained poses for ligands"""

        from openeye import oechem, oeomega

        # Make oechem be quiet
        oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Quiet)

        ref_mol = reference_ligand.to_oemol()
        # Prep the substructure searching
        ss = oechem.OESubSearch(core_smarts)
        oechem.OEPrepareSearch(ref_mol, ss)
        core_fragment = None

        for match in ss.Match(ref_mol):
            core_fragment = oechem.OEGraphMol()
            oechem.OESubsetMol(core_fragment, match)
            break

        if core_fragment is None:
            raise RuntimeError(
                f"A core fragment could not be extracted from the reference ligand using core smarts {core_smarts}"
            )

        # Create an Omega instance
        omega_opts = oeomega.OEOmegaOptions(oeomega.OEOmegaSampYEling_Dense)
        # Set the fixed reference molecule
        omega_fix_opts = oeomega.OEConfFixOptions()
        omega_fix_opts.SetFixMaxMatch(10)  # allow multiple MCSS matches
        omega_fix_opts.SetFixDeleteH(True)  # only use heavy atoms
        omega_fix_opts.SetFixMol(core_fragment)
        omega_fix_opts.SetFixRMS(1.0)
        # set the matching atom and bond expressions
        atomexpr = oechem.OEExprOpts_Aromaticity | oechem.OEExprOpts_AtomicNumber
        bondexpr = oechem.OEExprOpts_BondOrder | oechem.OEExprOpts_Aromaticity
        omega_fix_opts.SetAtomExpr(atomexpr)
        omega_fix_opts.SetBondExpr(bondexpr)
        omega_opts.SetConfFixOptions(omega_fix_opts)
        # set the builder options
        mol_builder_opts = oeomega.OEMolBuilderOptions()
        mol_builder_opts.SetStrictAtomTypes(
            False
        )  # don't give up if MMFF types are not found
        omega_opts.SetMolBuilderOptions(mol_builder_opts)
        omega_opts.SetWarts(False)  # expand molecule title
        omega_opts.SetStrictStereo(True)  # set strict stereochemistry
        omega_opts.SetIncludeInput(False)  # don't include input
        omega_opts.SetMaxConfs(self.max_confs)  # generate lots of conformers
        omega_opts.SetEnergyWindow(self.energy_window)  # allow high energies
        omega_generator = oeomega.OEOmega(omega_opts)

        # process the ligands
        result_ligands = []
        failed_ligands = []
        for mol in ligands:
            oe_mol = mol.to_oemol()
            # run omega
            return_code = omega_generator.Build(oe_mol)
            if (oe_mol.GetDimension() != 3) or (
                return_code != oeomega.OEOmegaReturnCode_Success
            ):
                # omega failed for this ligand, how do we track this?
                failed_ligands.append(Ligand.from_oemol(oe_mol))

            else:
                result_ligands.append(Ligand.from_oemol(oe_mol))


class PoseGeneratedLigands(BaseModel):
    """
    A basic results class to document the inputs and outputs of the docking.
    """

    type: Literal["PoseGenerationResult"] = "PoseGenerationResult"

    class Config:
        allow_mutation = False
        arbitrary_types_allowed = True

    pose_generator: OpenEyeConstrainedPoseGenerator = Field(
        ...,
        description="The pose generation engine and run time settings to generate the poses.",
    )
    provenance: dict[str, Any] = Field(
        ..., description="The provenance of the pose generation engine used."
    )
    receptor: PreppedTarget = Field(
        ..., description="The prepared receptor which was used in pose generation."
    )
    ligands: list[Ligand] = Field(
        ..., description="The list of ligands with their generated poses."
    )
    reference_ligand: ReferenceLigand = Field(
        ...,
        description="The reference ligand which is associated with the prepared target.",
    )
    failed_ligands: Optional[list[Ligand]] = Field(
        None, description="A list of ligands for which we failed to generate a pose."
    )
