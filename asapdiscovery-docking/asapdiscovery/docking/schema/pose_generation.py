import abc
from enum import Enum
from typing import Any, Literal, Optional

from asapdiscovery.data.backend.openeye import (
    get_SD_data,
    oechem,
    oedocking,
    oeff,
    oeomega,
    set_SD_data,
)
from asapdiscovery.data.schema.complex import PreppedComplex
from asapdiscovery.data.schema.ligand import Ligand
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt
from rdkit import Chem, RDLogger
import warnings
import multiprocessing.pool
import functools

RDLogger.DisableLog(
    "rdApp.*"
)  # disables some cpp-level warnings that can break multithreading


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


# Enums for the pose selectors
class PoseSelectionMethod(str, Enum):
    Chemgauss4 = "Chemgauss4"
    Chemgauss3 = "Chemgauss3"


class PoseEnergyMethod(str, Enum):
    MMFF = "MMFF"
    Sage = "Sage"
    Parsley = "Parsley"


class _BasicConstrainedPoseGenerator(BaseModel, abc.ABC):
    """An abstract class for other constrained pose generation methods to follow from."""

    type: Literal["_BasicConstrainedPoseGenerator"] = "_BasicConstrainedPoseGenerator"

    clash_cutoff: PositiveFloat = Field(
        2.0,
        description="The distance cutoff for which we check for clashes in Angstroms.",
    )
    selector: PoseSelectionMethod = Field(
        PoseSelectionMethod.Chemgauss3,
        description="The method which should be used to select the optimal conformer.",
    )
    backup_score: PoseEnergyMethod = Field(
        PoseEnergyMethod.Sage,
        description="If the main scoring function fails to descriminate between conformers the backup score will be used based on the internal energy of the molecule.",
    )

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
        processors: int = 1,
    ) -> tuple[list[oechem.OEMol], list[oechem.OEMol]]:
        """The main worker method which should generate ligand poses in the receptor using the reference ligand where required."""
        ...

    def generate_poses(
        self,
        prepared_complex: PreppedComplex,
        ligands: list[Ligand],
        core_smarts: Optional[str] = None,
        processors: int = 1,
    ) -> PosedLigands:
        """
        Generate poses for the given list of molecules in the target receptor.

        Note:
            We assume all stereo and states have been expanded and checked by this point.

        Args:
            prepared_complex: The prepared receptor and reference ligand which will be used to constrain the pose of the target ligands.
            ligands: The list of ligands which require poses in the target receptor.
            core_smarts: An optional smarts string which should be used to identify the MCS between the ligand and the reference, if not
                provided the MCS will be found using RDKit to preserve chiral centers.
            processors: The number of parallel process to use when generating the conformations.

        Returns:
            A list of ligands with new poses generated and list of ligands for which we could not generate a pose.
        """

        posed_ligands, failed_ligands = self._generate_poses(
            prepared_complex=prepared_complex,
            ligands=ligands,
            core_smarts=core_smarts,
            processors=processors,
        )
        # store the results, unpacking each posed conformer to a separate molecule
        result = PosedLigands()
        for oemol in posed_ligands:
            result.posed_ligands.append(Ligand.from_oemol(oemol))

        for fail_oemol in failed_ligands:
            result.failed_ligands.append(Ligand.from_oemol(fail_oemol))
        return result

    def _prune_clashes(self, receptor: oechem.OEMol, ligands: list[oechem.OEMol]):
        """
        Edit the conformers on the molecules in place to remove clashes with the receptor.

        Args:
            receptor: The receptor with which we should check for clashes.
            ligands: The list of ligands with conformers to prune.

        Returns:
            The ligands with clashed conformers removed.
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
        self, receptor: oechem.OEDesignUnit, ligands: list[oechem.OEMol]
    ) -> list[oechem.OEMol]:
        """
        Select the best pose for each ligand in place using the selected criteria.

        TODO split into separate methods once we have more selection options

        Args:
            receptor: The receptor oedu of the receptor with the binding site defined
            ligands: The list of multi-conformer ligands for which we want to select the best pose.

        Returns:
            A list of single conformer oe molecules with the optimal pose
        """
        scorers = {
            PoseSelectionMethod.Chemgauss4: oedocking.OEScoreType_Chemgauss4,
            PoseSelectionMethod.Chemgauss3: oedocking.OEScoreType_Chemgauss3,
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
                best_pose = self._select_by_energy(ligand)

            else:
                # set the best score as the active conformer
                poses = sorted(poses, key=lambda x: x[0])
                best_pose = oechem.OEGraphMol(poses[0][1])

                # set SD data to whole molecule, then get all the SD data and set to all conformers
                set_SD_data(
                    best_pose, {f"{self.selector.value}_score": str(poses[0][0])}
                )

            # turn back into a single conformer molecule
            posed_ligands.append(best_pose)
        return posed_ligands

    def _select_by_energy(self, ligand: oechem.OEMol) -> oechem.OEGraphMol:
        """
        Calculate the internal energy of each conformer of the ligand using the backup score force field and select the lowest energy pose as active

        Args:
            ligand: A multi-conformer OEMol we want to calculate the energies of.

        Notes:
            This edits the molecule in place.
        """
        force_fields = {
            PoseEnergyMethod.MMFF: oeff.OEMMFF,
            PoseEnergyMethod.Sage: oeff.OESage,
            PoseEnergyMethod.Parsley: oeff.OEParsley,
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
        best_pose = oechem.OEGraphMol(poses[0][1])
        set_SD_data(best_pose, {f"{self.backup_score.value}_energy": str(poses[0][0])})
        return best_pose


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

    def provenance(self) -> dict[str, Any]:
        return {
            "oechem": oechem.OEChemGetVersion(),
            "oeomega": oeomega.OEOmegaGetVersion(),
            "oedocking": oedocking.OEDockingGetVersion(),
            "oeff": oeff.OEFFGetVersion(),
        }

    def _generate_core_fragment(
        self, reference_ligand: oechem.OEMol, core_smarts: str
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

        # work with a copy and remove the hydrogens from the reference as we dont want to constrain to them
        input_mol = oechem.OEMol(reference_ligand)
        oechem.OESuppressHydrogens(input_mol)
        # build a query mol which allows for wild card matches
        # <https://github.com/choderalab/asapdiscovery/pull/430#issuecomment-1702360130>
        smarts_mol = oechem.OEGraphMol()
        oechem.OESmilesToMol(smarts_mol, core_smarts)
        pattern_query = oechem.OEQMol(smarts_mol)
        atomexpr = oechem.OEExprOpts_DefaultAtoms
        bondexpr = oechem.OEExprOpts_DefaultBonds
        pattern_query.BuildExpressions(atomexpr, bondexpr)
        ss = oechem.OESubSearch(pattern_query)
        oechem.OEPrepareSearch(input_mol, ss)
        core_fragment = None

        for match in ss.Match(input_mol):
            core_fragment = oechem.OEGraphMol()
            oechem.OESubsetMol(core_fragment, match)
            break

        if core_fragment is None:
            raise RuntimeError(
                f"A core fragment could not be extracted from the reference ligand using core smarts {core_smarts}"
            )
        return core_fragment

    def _generate_omega_instance(
        self, core_fragment: oechem.OEGraphMol, use_mcs: bool
    ) -> oeomega.OEOmega:
        """
        Create an instance of omega for constrained pose generation using the input core molecule and the runtime
        settings.

        Parameters
        ----------
        core_fragment: The OEGraphMol which should be used to define the constrained atoms during generation.
        use_mcs: If the core fragment is not defined by the user try and mcs match between it and the target ligands.

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
        if use_mcs:
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

    def _generate_pose(
        self,
        target_ligand: oechem.OEMol,
        reference_ligand: oechem.OEMol,
        core_smarts: Optional[str] = None,
    ) -> oechem.OEMol:
        """
        Use the configured openeye Omega instance to generate conformers for the target ligand.

        Args:
            target_ligand: The target ligand we want to generate the conformers for.
            reference_ligand: The ligand which should be used to restrain the target ligand conformers.
            core_smarts: The smarts which should be used to identify the mcs if not provided it will be determined automatically.

        Returns:
            The openeye molecule containing the posed conformers.
        """
        from asapdiscovery.data.backend.openeye import get_SD_data, set_SD_data

        if core_smarts is not None:
            core_fragment = self._generate_core_fragment(
                reference_ligand=reference_ligand, core_smarts=core_smarts
            )
            use_mcs = False
        else:
            # use the reference ligand and let openeye find the mcs match
            core_fragment = reference_ligand
            use_mcs = True

        # build and configure omega
        omega_generator = self._generate_omega_instance(
            core_fragment=core_fragment, use_mcs=use_mcs
        )

        # Get SD data because the omega code will silently move it to the high level
        # and that is inconsistent with what we do elsewhere
        sd_data = get_SD_data(target_ligand)

        # run omega
        return_code = omega_generator.Build(target_ligand)

        # deal with strange hydrogen DO NOT REMOVE
        oechem.OESuppressHydrogens(target_ligand)
        oechem.OEAddExplicitHydrogens(target_ligand)

        # add SD data back
        set_SD_data(target_ligand, sd_data)

        if (target_ligand.GetDimension() != 3) or (
            return_code != oeomega.OEOmegaReturnCode_Success
        ):
            # add the failure message as an SD tag, should be able to see visually if the molecule is 2D

            set_SD_data(
                mol=target_ligand,
                data={"omega_return_code": oeomega.OEGetOmegaError(return_code)},
            )
        return target_ligand

    def _generate_poses(
        self,
        prepared_complex: PreppedComplex,
        ligands: list[Ligand],
        core_smarts: Optional[str] = None,
        processors: int = 1,
    ) -> tuple[list[oechem.OEMol], list[oechem.OEMol]]:
        """
        Use openeye oeomega to generate constrained poses for the input ligands. The core smarts is used to decide
        which atoms should be constrained if not supplied the MCS will be found by openeye.
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed

        from tqdm import tqdm

        # Make oechem be quiet
        oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Quiet)

        # grab the reference ligand
        reference_ligand = prepared_complex.ligand.to_oemol()

        # process the ligands
        result_ligands = []
        failed_ligands = []

        if processors > 1:
            progressbar = tqdm(total=len(ligands))
            with ProcessPoolExecutor(max_workers=processors) as pool:
                work_list = [
                    pool.submit(
                        self._generate_pose,
                        **{
                            "target_ligand": mol.to_oemol(),
                            "core_smarts": core_smarts,
                            "reference_ligand": reference_ligand,
                        },
                    )
                    for mol in ligands
                ]
                for work in as_completed(work_list):
                    target_ligand = work.result()
                    # check if coordinates could be generated
                    if "omega_return_code" in get_SD_data(target_ligand):
                        failed_ligands.append(target_ligand)
                    else:
                        result_ligands.append(target_ligand)
                    progressbar.update(1)
        else:
            for mol in tqdm(ligands, total=len(ligands)):
                posed_ligand = self._generate_pose(
                    target_ligand=mol.to_oemol(),
                    core_smarts=core_smarts,
                    reference_ligand=reference_ligand,
                )
                # check if coordinates could be generated
                if "omega_return_code" in get_SD_data(posed_ligand):
                    failed_ligands.append(posed_ligand)
                else:
                    result_ligands.append(posed_ligand)

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


class RDKitConstrainedPoseGenerator(_BasicConstrainedPoseGenerator):
    """Use RDKit to embed multiple conformers of the molecule while constraining it to the template ligand."""

    type: Literal["RDKitConstrainedPoseGenerator"] = "RDKitConstrainedPoseGenerator"

    max_confs: PositiveInt = Field(
        300, description="The maximum number of conformers to try and generate."
    )
    rms_thresh: PositiveFloat = Field(
        0.2,
        description="Retain only the conformations out of 'numConfs' after embedding that are at least this far apart from each other. RMSD is computed on the heavy atoms.",
    )
    mcs_timeout: PositiveInt = Field(
        1, description="The timeout in seconds to run the mcs search in RDKit."
    )

    def provenance(self) -> dict[str, Any]:
        import openff.toolkit
        import rdkit

        return {
            "oechem": oechem.OEChemGetVersion(),
            "oeff": oeff.OEFFGetVersion(),
            "oedocking": oedocking.OEDockingGetVersion(),
            "rdkit": rdkit.__version__,
            "openff.toolkit": openff.toolkit.__version__,
        }

    def _generate_mcs_core(
        self, target_ligand: Chem.Mol, reference_ligand: Chem.Mol
    ) -> Chem.Mol:
        """
        For the given target and reference ligand find an MCS match to generate
        a new template ligand which can be used in the constrained embedding.

        Args:
            target_ligand: The target ligand we want to generate the pose for.
            reference_ligand: The reference ligand which we want to find the mcs overlap with.

        Returns:
            An rdkit molecule created from the MCS overlap of the two ligands.
        """
        from rdkit import Chem
        from rdkit.Chem import rdFMCS

        mcs = rdFMCS.FindMCS(
            [target_ligand, reference_ligand],
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            atomCompare=rdFMCS.AtomCompare.CompareAnyHeavyAtom,
            bondCompare=rdFMCS.BondCompare.CompareAny,
            maximizeBonds=False,
            timeout=self.mcs_timeout,
        )
        return Chem.MolFromSmarts(mcs.smartsString)

    def _transfer_coordinates(
        self, reference_ligand: Chem.Mol, template_ligand: Chem.Mol
    ) -> Chem.Mol:
        """
        Transfer the coordinates from the reference to the template ligand.

        Args:
            reference_ligand: The ligand we want to generate the conformers for.
            template_ligand: The ligand whose coordinates should be used as a reference.

        Returns:
            The template ligand with atom positions set to the reference for overlapping atoms.

        """
        matches = reference_ligand.GetSubstructMatch(template_ligand)
        if not matches:
            raise RuntimeError(
                f"A core fragment could not be extracted from the reference ligand using core smarts {Chem.MolToSmarts(template_ligand)}"
            )

        ref_conformer: Chem.Conformer = reference_ligand.GetConformer(0)
        template_conformer = Chem.Conformer()
        for i, atom_match in enumerate(matches):
            ref_atom_position = ref_conformer.GetAtomPosition(atom_match)
            template_conformer.SetAtomPosition(i, ref_atom_position)
        template_ligand.AddConformer(template_conformer, assignId=True)
        return template_ligand

    def _generate_coordinate_map(
        self, target_ligand: Chem.Mol, template_ligand: Chem.Mol
    ) -> tuple[dict, list]:
        """
        Generate a mapping between the target ligand atom index and the reference atoms coordinates.

        Args:
            target_ligand: The ligand we want to generate the conformers for.
            template_ligand: The ligand whose coordinates should be used as a reference.

        Returns:
            A tuple contacting a dictionary which maps the target ligand indices to a reference atom coordinate and a
            list of tuples matching the target and template ligand atom indices for any equivalent atoms.

        """
        # map the scaffold atoms to the new molecule
        # we assume the template has a single conformer
        template_conformer = template_ligand.GetConformer(0)
        match = target_ligand.GetSubstructMatch(template_ligand)
        coords_map = {}
        index_map = []
        for core_index, matched_index in enumerate(match):
            core_atom_coord = template_conformer.GetAtomPosition(core_index)
            coords_map[matched_index] = core_atom_coord
            index_map.append((matched_index, core_index))

        return coords_map, index_map

    def timeout(max_timeout):
        """Timeout decorator, parameter in seconds."""

        def timeout_decorator(item):
            """Wrap the original function."""

            @functools.wraps(item)
            def func_wrapper(*args, **kwargs):
                """Closure for function."""
                pool = multiprocessing.pool.ThreadPool(processes=1)
                async_result = pool.apply_async(item, args, kwargs)
                # raises a TimeoutError if execution exceeds max_timeout
                return async_result.get(max_timeout)

            return func_wrapper

        return timeout_decorator

    @timeout(0.2)
    def _generate_pose(
        self,
        target_ligand: Chem.Mol,
        core_ligand: Chem.Mol,
        core_smarts: Optional[str] = None,
    ) -> Chem.Mol:
        """
        Generate the poses for the target molecule while restraining the MCS to the core ligand.

        Args:
            target_ligand: The ligand we wish to generate the MCS restrained poses for.
            core_ligand: The reference ligand whose coordinates we should match.
            core_smarts: The smarts pattern which should be used to define the mcs between the target and the core ligand.

        Returns:
            An rdkit molecule with the generated poses to be filtered.

        Note:
            This function always returns a molecules even if generation fails it will just have no conformations.
        """

        from rdkit.Chem import AllChem  # noqa needed to trigger force fields in rdkit
        from rdkit.Chem.rdDistGeom import EmbedMultipleConfs
        from rdkit.Chem.rdForceFieldHelpers import UFFGetMoleculeForceField
        from rdkit.Chem.rdMolAlign import AlignMol

        # run to make sure we don't lose molecule properties when using pickle
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

        if core_smarts is not None:
            # extract the template mol based on this core smarts
            template_mol = self._generate_mcs_core(
                target_ligand=Chem.MolFromSmiles(core_smarts),
                reference_ligand=core_ligand,
            )
        else:
            # use mcs to find the template mol
            template_mol = self._generate_mcs_core(
                target_ligand=target_ligand, reference_ligand=core_ligand
            )
        # transfer the relevant coordinates from the crystal core to the template
        template_mol = self._transfer_coordinates(
            reference_ligand=core_ligand, template_ligand=template_mol
        )
        # create a coordinate and atom index map for the embedding
        coord_map, index_map = self._generate_coordinate_map(
            target_ligand=target_ligand, template_ligand=template_mol
        )
        # embed multiple conformers
        embeddings = list(
            EmbedMultipleConfs(
                target_ligand,
                numConfs=self.max_confs,
                clearConfs=True,
                pruneRmsThresh=self.rms_thresh,
                coordMap=coord_map,
                enforceChirality=True,
                useExpTorsionAnglePrefs=True,
                useBasicKnowledge=True,
                useSmallRingTorsions=True,
            )
        )
        if len(embeddings) != 0:
            for embedding in embeddings:
                _ = AlignMol(
                    target_ligand, template_mol, prbCid=embedding, atomMap=index_map
                )

                # TODO expose MMFF as an option
                ff = UFFGetMoleculeForceField(target_ligand, confId=embedding)
                conf = template_mol.GetConformer()
                for matched_index, core_index in index_map:
                    coord = conf.GetAtomPosition(core_index)
                    coord_index = (
                        ff.AddExtraPoint(coord.x, coord.y, coord.z, fixed=True) - 1
                    )
                    ff.AddDistanceConstraint(
                        coord_index, matched_index, 0, 0, 100.0 * 100
                    )

                ff.Initialize()
                n = 4
                more = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
                while more and n:
                    more = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
                    n -= 1

                # realign
                _ = AlignMol(
                    target_ligand, template_mol, prbCid=embedding, atomMap=index_map
                )

        return target_ligand

    def work(self, target_ligand, core_ligand, core_smarts):
        """
        Generates a pose for a target_ligand while wrapping the whole thing in a try-except
        so we can catch edge-cases. This enables multi-processed pose generation to return
        exceptions gracefully.

        Returns a success bool, the posed ligand (or input ligand in case of fail) and the
        error message.
        """
        try:
            return (
                True,
                self._generate_pose(
                    target_ligand=target_ligand.to_rdkit(),
                    core_ligand=core_ligand,
                    core_smarts=core_smarts,
                ),
                None,
            )
        except Exception as e:
            return False, target_ligand, e

    def _generate_poses(
        self,
        prepared_complex: PreppedComplex,
        ligands: list[Ligand],
        core_smarts: Optional[str] = None,
        processors: int = 1,
    ) -> tuple[list[oechem.OEMol], list[oechem.OEMol]]:
        """
        Use RDKit to embed multiple conformers which are constrained to the template molecule.

        Args:
            prepared_complex: The reference complex containing the receptor and small molecule which has been prepared.
            ligands: The list of ligands to generate poses for.
            core_smarts: The core smarts which should be used to define the core molecule.
            processors: The number of processes to use when generating the conformations.

        Returns:
            Two lists the first of the successfully posed ligands and ligands which failed.
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed

        from openff.toolkit import Molecule
        from tqdm import tqdm

        # make sure we are not using hs placed by prep as a reference coordinate for the generated conformers
        core_ligand = Chem.RemoveHs(prepared_complex.ligand.to_rdkit())

        # setup the rdkit pickle properties to save all molecule properties
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

        # process the ligands
        result_ligands = []
        failed_ligands = []

        if processors > 1:
            progressbar = tqdm(total=len(ligands))
            with ProcessPoolExecutor(max_workers=processors) as pool:
                work_list = [
                    pool.submit(
                        self.work,
                        **{
                            "target_ligand": mol,
                            "core_ligand": core_ligand,
                            "core_smarts": core_smarts,
                        },
                    )
                    for mol in ligands
                ]
                for work in as_completed(work_list):

                    succ, target_ligand, err_code = work.result()
                    if succ:
                        try:
                            off_mol = Molecule.from_rdkit(
                                target_ligand, allow_undefined_stereo=True
                            )
                            # we need to transfer the properties which would be lost
                            openeye_mol = off_mol.to_openeye()

                            # make sure properties at the top level get added to the conformers
                            sd_tags = get_SD_data(openeye_mol)
                            set_SD_data(openeye_mol, sd_tags)

                            if target_ligand.GetNumConformers() > 0:
                                # save the mol with all conformers
                                result_ligands.append(openeye_mol)
                            else:
                                failed_ligands.append(openeye_mol)
                        except Exception as e:
                            warnings.warn(
                                f"Ligand posing failed for ligand {Chem.MolToSmiles(target_ligand)} with exception: {e}"
                            )
                    else:
                        warnings.warn(
                            f"Ligand posing failed for ligand {target_ligand.smiles} with exception: {err_code}"
                        )

                    progressbar.update(1)
        else:
            for mol in tqdm(ligands, total=len(ligands)):
                # try:
                posed_ligand = self._generate_pose(
                    target_ligand=Chem.AddHs(mol.to_rdkit()),
                    core_ligand=core_ligand,
                    core_smarts=core_smarts,
                )

                off_mol = Molecule.from_rdkit(posed_ligand, allow_undefined_stereo=True)
                # we need to transfer the properties which would be lost
                openeye_mol = off_mol.to_openeye()

                # make sure properties at the top level get added to the conformers
                sd_tags = get_SD_data(openeye_mol)
                set_SD_data(openeye_mol, sd_tags)

                if posed_ligand.GetNumConformers() > 0:
                    # save the mol with all conformers
                    result_ligands.append(openeye_mol)
                else:
                    failed_ligands.append(openeye_mol)
                # except Exception as e:
                #     warnings.warn(
                #         f"Ligand posing failed for ligand {mol.compound_name}:{mol.smiles} with exception: {e}"
                #     )
                #     failed_ligands.append(mol.to_oemol())

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
