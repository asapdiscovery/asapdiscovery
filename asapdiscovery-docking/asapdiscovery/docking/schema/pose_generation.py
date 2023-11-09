import abc
from typing import Any, Literal, Optional

from asapdiscovery.data.openeye import oechem, oedocking, oeff, oeomega, set_SD_data
from asapdiscovery.data.schema_v2.complex import PreppedComplex
from asapdiscovery.data.schema_v2.ligand import Ligand
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt


class PosedLigands(BaseModel):
    """
    A results class to handle the posed and failed ligands.
    """

    posed_ligands: list[Ligand] = Field(
        [], description="A list of Ligands with a single final pose."
    )
    failed_ligands: list[Ligand] = Field(
        [], description="The list of Ligands which failed the pose generation stage."
    )


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
        prepared_complex: PreppedComplex,
        ligands: list[Ligand],
        core_smarts: Optional[str] = None,
    ) -> tuple[list[oechem.OEMol], list[oechem.OEMol]]:
        """The main worker method which should generate ligand poses in the receptor using the reference ligand where required."""
        ...

    def generate_poses(
        self,
        prepared_complex: PreppedComplex,
        ligands: list[Ligand],
        core_smarts: Optional[str],
    ) -> PosedLigands:
        """
        Generate poses for the given list of molecules in the target receptor.

        Note:
            We assume all stereo and states have been expanded and checked by this point.

        Parameters
        ----------
        prepared_complex: The prepared receptor and reference ligand which will be used to constrain the pose of the target ligands.
        ligands: The list of ligands which require poses in the target receptor.
        core_smarts: An optional smarts string which should be used to identify the MCS between the ligand and the reference, if not
        provided the MCS will be found using RDKit to preserve chiral centers.

        Returns
        -------
            A list of ligands with new poses generated and list of ligands for which we could not generate a pose.
        """

        posed_ligands, failed_ligands = self._generate_poses(
            prepared_complex=prepared_complex,
            ligands=ligands,
            core_smarts=core_smarts,
        )
        # store the results, unpacking each posed conformer to a separate molecule
        result = PosedLigands()
        for oemol in posed_ligands:
            result.posed_ligands.append(Ligand.from_oemol(oemol))

        for fail_oemol in failed_ligands:
            result.failed_ligands.append(Ligand.from_oemol(fail_oemol))

        return result


class OpenEyeConstrainedPoseGenerator(_BasicConstrainedPoseGenerator):
    type: Literal["OpenEyeConstrainedPoseGenerator"] = "OpenEyeConstrainedPoseGenerator"
    max_confs: PositiveInt = Field(
        1000, description="The maximum number of conformers to try and generate."
    )
    energy_window: PositiveFloat = Field(
        20,
        description="Sets the maximum allowable energy difference between the lowest and the highest energy conformers,"
        " in units of kcal/mol.",
    )
    clash_cutoff: PositiveFloat = Field(
        2.0,
        description="The distance cutoff for which we check for clashes in Angstroms.",
    )
    selector: Literal["Chemgauss4", "Chemgauss3"] = Field(
        "Chemgauss3",
        description="The method which should be used to select the optimal conformer.",
    )
    backup_score: Literal["MMFF", "Sage", "Parsley"] = Field(
        "Sage",
        description="If the main scoring function fails to descriminate between conformers the backup score will be used based on the internal energy of the molecule.",
    )

    def provenance(self) -> dict[str, Any]:
        from openeye import oechem, oeff, oeomega

        return {
            "oechem": oechem.OEChemGetVersion(),
            "oeomega": oeomega.OEOmegaGetVersion(),
            "oedocking": oedocking.OEDockingGetVersion(),
            "oeff": oeff.OEFFGetVersion(),
        }

    def _generate_core_fragment(
        self, reference_ligand: Ligand, core_smarts: str
    ) -> oechem.OEGraphMol:
        """
        Generate an openeye GraphMol of the core fragment made from the MCS match between the ligand and core smarts
        which will be used to constrain the geometries of the ligands during pose generation.

        Parameters
        ----------
        reference_ligand: The ligand whose pose we will be constrained to match.
        core_smarts: The SMARTS pattern used to identify the MCS in the reference ligand.

        Returns
        -------
            An OEGraphMol of the MCS matched core fragment.
        """

        ref_mol = reference_ligand.to_oemol()
        # build a query mol which allows for wild card matches
        # <https://github.com/choderalab/asapdiscovery/pull/430#issuecomment-1702360130>
        smarts_mol = oechem.OEGraphMol()
        oechem.OESmilesToMol(smarts_mol, core_smarts)
        pattern_query = oechem.OEQMol(smarts_mol)
        atomexpr = oechem.OEExprOpts_DefaultAtoms
        bondexpr = oechem.OEExprOpts_DefaultBonds
        pattern_query.BuildExpressions(atomexpr, bondexpr)
        ss = oechem.OESubSearch(pattern_query)
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
        return core_fragment

    def _generate_omega_instance(
        self, core_fragment: oechem.OEGraphMol
    ) -> oeomega.OEOmega:
        """
        Create an instance of omega for constrained pose generation using the input core molecule and the runtime
        settings.

        Parameters
        ----------
        core_fragment: The OEGraphMol which should be used to define the constrained atoms during generation.

        Returns
        -------
            An instance of omega configured for the current run.
        """

        # Create an Omega instance
        omega_opts = oeomega.OEOmegaOptions(oeomega.OEOmegaSampling_Dense)
        # Set the fixed reference molecule
        omega_fix_opts = oeomega.OEConfFixOptions()
        omega_fix_opts.SetFixMaxMatch(10)  # allow multiple MCSS matches
        omega_fix_opts.SetFixDeleteH(True)  # only use heavy atoms
        omega_fix_opts.SetFixMol(core_fragment)  # Provide the reference ligand
        omega_fix_opts.SetFixMCS(True)
        omega_fix_opts.SetFixRMS(
            1.0
        )  # The maximum distance between two atoms which is considered identical
        # set the matching atom and bond expressions
        atomexpr = (
            oechem.OEExprOpts_Aromaticity
            | oechem.OEExprOpts_AtomicNumber
            | oechem.OEExprOpts_RingMember
        )
        bondexpr = oechem.OEExprOpts_Aromaticity | oechem.OEExprOpts_RingMember
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

        return omega_generator

    def _generate_poses(
        self,
        prepared_complex: PreppedComplex,
        ligands: list[Ligand],
        core_smarts: Optional[str] = None,
    ) -> tuple[list[oechem.OEMol], list[oechem.OEMol]]:
        """
        Use openeye oeomega to generate constrained poses for the input ligands. The core smarts is used to decide
        which atoms should be constrained if not supplied the MCS will be found by openeye.
        """

        # Make oechem be quiet
        oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Quiet)

        # generate a core fragment used to constrain the generated poses
        if core_smarts:
            core_fragment = self._generate_core_fragment(
                reference_ligand=prepared_complex.ligand, core_smarts=core_smarts
            )
        else:
            # use the reference ligand and let openeye find the mcs match
            core_fragment = prepared_complex.ligand.to_oemol()

        # build and configure omega
        omega_generator = self._generate_omega_instance(core_fragment=core_fragment)

        # process the ligands
        result_ligands = []
        failed_ligands = []
        for mol in ligands:
            oe_mol = oechem.OEMol(mol.to_oemol())
            # run omega
            return_code = omega_generator.Build(oe_mol)

            # deal with strange hydrogen DO NOT REMOVE
            oechem.OESuppressHydrogens(oe_mol)
            oechem.OEAddExplicitHydrogens(oe_mol)

            if (oe_mol.GetDimension() != 3) or (
                return_code != oeomega.OEOmegaReturnCode_Success
            ):
                # add the failure message as an SD tag, should be able to see visually if the molecule is 2D
                set_SD_data(
                    mol=oe_mol,
                    data={"omega_return_code": oeomega.OEGetOmegaError(return_code)},
                )
                # omega failed for this ligand, how do we track this?
                failed_ligands.append(oe_mol)

            else:
                result_ligands.append(oe_mol)

        # prue down the conformers
        oedu_receptor = prepared_complex.target.to_oedu()
        oe_receptor = oechem.OEGraphMol()
        oedu_receptor.GetProtein(oe_receptor)

        self._prune_clashes(receptor=oe_receptor, ligands=result_ligands)
        # select the best pose to be kept
        posed_ligands = self._select_best_pose(
            receptor=oedu_receptor, ligands=result_ligands
        )

        return posed_ligands, failed_ligands

    def _prune_clashes(self, receptor: oechem.OEMol, ligands: list[oechem.OEMol]):
        """
        Edit the conformers on the molecules in place to remove clashes with the receptor.

        Parameters
        ----------
        receptor: The receptor with which we should check for clashes.
        ligands: The list of ligands with conformers to prune.

        """
        import numpy as np

        # setup the function to check for close neighbours
        near_nbr = oechem.OENearestNbrs(receptor, self.clash_cutoff)

        for ligand in ligands:
            if ligand.NumConfs() < 10:
                # only filter if we have more than 10 confs
                continue

            poses = []
            for conformer in ligand.GetConfs():
                clash_score = 0
                for nb in near_nbr.GetNbrs(conformer):
                    if (not nb.GetBgn().IsHydrogen()) and (
                        not nb.GetEnd().IsHydrogen()
                    ):
                        # use an exponentially decaying penalty on each distance below the cutoff not between hydrogen
                        clash_score += np.exp(
                            -0.5 * (nb.GetDist() / self.clash_cutoff) ** 2
                        )

                poses.append((clash_score, conformer))
            # eliminate the worst 50% of clashes
            poses = sorted(poses, key=lambda x: x[0])
            for _, conformer in poses[int(0.5 * len(poses)) :]:
                ligand.DeleteConf(conformer)

    def _select_best_pose(
        self, receptor: oechem.OEMol, ligands: list[oechem.OEMol]
    ) -> list[oechem.OEGraphMol]:
        """
        Select the best pose for each ligand in place using the selected criteria.

        TODO split into separate methods once we have more selection options

        Parameters
        ----------
        receptor: The receptor the ligands poses should be scored against.
        ligands: The list of multi-conformer ligands for which we want to select the best pose.

        Returns
        -------
        A list of single conformer oe molecules with the optimal pose

        """
        scorers = {
            "Chemgauss4": oedocking.OEScoreType_Chemgauss4,
            "Chemgauss3": oedocking.OEScoreType_Chemgauss3,
        }
        score = oedocking.OEScore(scorers[self.selector])
        score.Initialize(receptor)
        posed_ligands = []
        for ligand in ligands:
            poses = [
                (score.ScoreLigand(conformer), conformer)
                for conformer in ligand.GetConfs()
            ]

            # check that the scorer worked else call the backup
            # this will select the lowest energy conformer
            unique_scores = {pose[0] for pose in poses}
            if len(unique_scores) == 1 and len(poses) != 1:
                self._select_by_energy(ligand)

            else:
                # set the best score as the active conformer
                poses = sorted(poses, key=lambda x: x[0])
                ligand.SetActive(poses[0][1])
                oechem.OESetSDData(ligand, f"{self.selector}_score", str(poses[0][0]))

            # turn back into a single conformer molecule
            posed_ligands.append(oechem.OEGraphMol(ligand))

        return posed_ligands

    def _select_by_energy(self, ligand: oechem.OEMol):
        """
        Calculate the internal energy of each conformer of the ligand using the backup score force field and select the lowest energy pose as active

        Parameters
        ----------
        ligand: A multi-conformer OEMol we want to calculate the energies of.

        """
        force_fields = {
            "MMFF": oeff.OEMMFF,
            "Sage": oeff.OESage,
            "Parsley": oeff.OEParsley,
        }
        ff = force_fields[self.backup_score]()
        ff.PrepMol(ligand)
        ff.Setup(ligand)
        vec_coords = oechem.OEDoubleArray(3 * ligand.GetMaxAtomIdx())
        poses = []
        for conformer in ligand.GetConfs():
            conformer.GetCoords(vec_coords)
            poses.append((ff(vec_coords), conformer))

        poses = sorted(poses, key=lambda x: x[0])
        ligand.SetActive(poses[0][1])
        oechem.OESetSDData(ligand, f"{self.backup_score}_energy", str(poses[0][0]))
