"""
This module contains the inputs, docker, and output schema for using POSIT
"""

import logging
from pathlib import Path
from typing import ClassVar, Literal, Optional, Union

import pandas as pd
from asapdiscovery.data.backend.openeye import oechem, oedocking, oeomega
from asapdiscovery.data.schema.ligand import Ligand
from asapdiscovery.data.util.dask_utils import dask_vmap
from asapdiscovery.data.util.intenum import IntEnum
from asapdiscovery.docking.docking import (
    DockingBase,
    DockingInputBase,
    DockingInputMultiStructure,
    DockingInputPair,
    DockingResult,
)
from asapdiscovery.docking.docking_data_validation import DockingResultCols
from pydantic.v1 import Field, PositiveInt, root_validator

logger = logging.getLogger(__name__)


class POSIT_METHOD(IntEnum):
    """
    Enum for POSIT methods
    """

    ALL = oedocking.OEPositMethod_ALL
    HYBRID = oedocking.OEPositMethod_HYBRID
    FRED = oedocking.OEPositMethod_FRED
    # this is a fake method but it is in the docs
    # MCS = oedocking.OEPositMethod_MCS
    SHAPEFIT = oedocking.OEPositMethod_SHAPEFIT


class POSIT_RELAX_MODE(IntEnum):
    """
    Enum for POSIT relax modes
    """

    CLASH = oedocking.OEPoseRelaxMode_CLASHED
    ALL = oedocking.OEPoseRelaxMode_ALL
    NONE = oedocking.OEPoseRelaxMode_NONE


class POSITDockingResults(DockingResult):
    """
    Schema for a DockingResult from OEPosit, containing both a DockingInputPair used as input to the workflow
    and a Ligand object containing the docked pose.
    """

    type: Literal["POSITDockingResults"] = "POSITDockingResults"

    def _get_single_pose_results(self) -> list["POSITDockingResults"]:
        """
        Since Ligand can contain multiple poses, but our scoring methods don't (as of 2024.04.24) support
        multiple poses, it's useful to have a method that can get single pose docking results.
        """
        return [
            POSITDockingResults(
                input_pair=self.input_pair,
                posed_ligand=lig,
                probability=lig.tags[DockingResultCols.DOCKING_CONFIDENCE_POSIT.value],
                provenance=self.provenance,
                pose_id=lig.tags["Pose_ID"],
            )
            for lig in self.posed_ligand.to_single_conformers()
        ]

    @staticmethod
    def make_df_from_docking_results(results: list["DockingResult"]):
        """
        Make a dataframe from a list of DockingResults

        Parameters
        ----------
        results : list[DockingResult]
            List of DockingResults

        Returns
        -------
        pd.DataFrame
            Dataframe of results
        """
        import pandas as pd

        df_prep = []
        for result in results:
            docking_dict = {}
            docking_dict[DockingResultCols.LIGAND_ID.value] = (
                result.input_pair.ligand.compound_name
            )
            docking_dict[DockingResultCols.TARGET_ID.value] = (
                result.input_pair.complex.target.target_name
            )
            docking_dict["target_bound_compound_smiles"] = (
                result.input_pair.complex.ligand.smiles
            )
            docking_dict[DockingResultCols.SMILES.value] = (
                result.input_pair.ligand.smiles
            )
            docking_dict[DockingResultCols.DOCKING_CONFIDENCE_POSIT.value] = (
                result.probability
            )
            df_prep.append(docking_dict)

        df = pd.DataFrame(df_prep)
        return df

    def to_df(self) -> pd.DataFrame:
        """
        Make a dataframe from a DockingResult

        Returns
        -------
        pd.DataFrame
            Dataframe of results
        """
        return self.make_df_from_docking_results([self])


class POSITDocker(DockingBase):
    type: Literal["POSITDocker"] = "POSITDocker"

    result_cls: ClassVar[POSITDockingResults] = POSITDockingResults

    relax_mode: POSIT_RELAX_MODE = Field(
        POSIT_RELAX_MODE.NONE,
        description="When to check for relaxation either, 'clash', 'all', 'none'",
    )
    posit_method: POSIT_METHOD = Field(
        POSIT_METHOD.ALL, description="POSIT method to use"
    )
    use_omega: bool = Field(True, description="Use omega to generate conformers")
    omega_dense: bool = Field(
        False, description="Use dense conformer generation with omega"
    )
    num_poses: PositiveInt = Field(1, description="Number of poses to generate")
    allow_low_posit_prob: bool = Field(True, description="Allow low posit probability")
    low_posit_prob_thresh: float = Field(
        0.0,
        description="Minimum posit probability threshold if allow_low_posit_prob is False",
    )
    allow_final_clash: bool = Field(
        False, description="Allow clashing poses in last stage of docking"
    )
    allow_retries: bool = Field(
        True,
        description="Allow retries with different options if docking fails initially",
    )
    last_ditch_fred: bool = Field(
        False, description="Use pure FRED docking as a last ditch effort"
    )

    @root_validator
    @classmethod
    def omega_dense_check(cls, values):
        """
        Validate omega_dense
        """
        omega_dense = values.get("omega_dense")
        use_omega = values.get("use_omega")
        if omega_dense and not use_omega:
            raise ValueError("Cannot use omega_dense without use_omega")
        return values

    @staticmethod
    def to_result_type():
        return POSITDockingResults

    @staticmethod
    def run_oe_posit_docking(
        opts, pose_res, dus: list[oechem.OEDesignUnit], lig, num_poses
    ):
        """
        Helper function to run OEPosit docking

        Parameters
        ----------
        opts : oedocking.OEPositOptions
            OEPosit options
        pose_res : oedocking.OEPositResults
            OEPosit results
        dus : list[oedocking.OEDesignUnit]
            OEDesignUnit
        lig : oechem.OEMol
            Ligand
        num_poses : int
            Number of poses to generate

        Returns
        -------
        oedocking.OEPositResults
            OEPosit results
        int
            Return code
        """
        poser = oedocking.OEPosit(opts)
        retcodes = [poser.AddReceptor(du) for du in dus]
        if not all(retcodes):
            raise ValueError("Failed to add receptor(s) to POSIT")
        retcode = poser.Dock(pose_res, lig, num_poses)
        return pose_res, retcode

    @dask_vmap(["inputs"], has_failure_mode=True)
    def _dock(
        self,
        inputs: list[DockingInputBase],
        output_dir: Optional[Union[str, Path]] = None,
        failure_mode="skip",
        return_for_disk_backend=False,
        **kwargs,
    ) -> list[DockingResult]:
        """
        Docking workflow using OEPosit
        """
        if output_dir is None and return_for_disk_backend:
            raise ValueError(
                "Cannot specify return_for_disk_backend and not output_dir"
            )

        docking_results = []

        for set in inputs:
            try:
                # make sure its a path
                output_dir = Path(output_dir) if output_dir is not None else None

                if output_dir is not None:
                    docked_result_path = Path(Path(output_dir) / set.unique_name)

                    jsons = list(
                        docked_result_path.glob("docking_result_*.json")
                    )  # can be multiple poses

                # first check if output exists
                if set.is_cacheable and (output_dir is not None) and (len(jsons) > 0):
                    logger.info(
                        f"Docking result for {set.unique_name} already exists, reading from disk"
                    )
                    for docked_result_json_path in jsons:
                        if return_for_disk_backend:
                            docking_results.append(docked_result_json_path)
                        else:
                            docking_results.append(
                                POSITDockingResults.from_json_file(
                                    docked_result_json_path
                                )
                            )
                # run docking if output does not exist
                else:
                    dus = set.to_design_units()
                    lig_oemol = oechem.OEMol(set.ligand.to_oemol())
                    if self.use_omega:
                        if self.omega_dense:
                            omegaOpts = oeomega.OEOmegaOptions(
                                oeomega.OEOmegaSampling_Dense
                            )
                        else:
                            omegaOpts = oeomega.OEOmegaOptions()
                        # set stereochemistry to non-strict
                        omegaOpts.SetStrictStereo(False)
                        omega = oeomega.OEOmega(omegaOpts)
                        omega_retcode = omega.Build(lig_oemol)
                        if omega_retcode:
                            error_msg = f"Omega failed with error code: {omega_retcode} : {oeomega.OEGetOmegaError(omega_retcode)}"
                            if failure_mode == "skip":
                                logger.error(error_msg)
                            elif failure_mode == "raise":
                                raise ValueError(error_msg)
                            else:
                                raise ValueError(
                                    f"Unknown error handling option {failure_mode}"
                                )

                    opts = oedocking.OEPositOptions()
                    opts.SetIgnoreNitrogenStereo(True)
                    opts.SetPositMethods(self.posit_method.value)
                    opts.SetPoseRelaxMode(self.relax_mode.value)

                    pose_res = oedocking.OEPositResults()
                    pose_res, retcode = self.run_oe_posit_docking(
                        opts, pose_res, dus, lig_oemol, self.num_poses
                    )

                    if self.allow_retries:
                        # try again with no relaxation
                        if (
                            retcode
                            == oedocking.OEDockingReturnCode_NoValidNonClashPoses
                        ):
                            opts.SetPoseRelaxMode(oedocking.OEPoseRelaxMode_NONE)
                            pose_res, retcode = self.run_oe_posit_docking(
                                opts, pose_res, dus, lig_oemol, self.num_poses
                            )

                        # try again with low posit probability
                        if (
                            retcode
                            == oedocking.OEDockingReturnCode_NoValidNonClashPoses
                            and self.allow_low_posit_prob
                        ):
                            opts.SetPoseRelaxMode(oedocking.OEPoseRelaxMode_ALL)
                            opts.SetMinProbability(self.low_posit_prob_thresh)
                            pose_res, retcode = self.run_oe_posit_docking(
                                opts, pose_res, dus, lig_oemol, self.num_poses
                            )

                        # try again allowing clashes
                        if (
                            self.allow_final_clash
                            and retcode
                            == oedocking.OEDockingReturnCode_NoValidNonClashPoses
                        ):
                            opts.SetPoseRelaxMode(oedocking.OEPoseRelaxMode_ALL)
                            opts.SetAllowedClashType(oedocking.OEAllowedClashType_ANY)
                            pose_res, retcode = self.run_oe_posit_docking(
                                opts, pose_res, dus, lig_oemol, self.num_poses
                            )

                    # try again with FRED
                    if (
                        retcode != oedocking.OEDockingReturnCode_Success
                        and self.last_ditch_fred
                    ):
                        opts_fred = oedocking.OEPositOptions()
                        opts_fred.SetIgnoreNitrogenStereo(True)
                        opts_fred.SetPositMethods(POSIT_METHOD.FRED)
                        opts_fred.SetPoseRelaxMode(self.relax_mode.value)
                        pose_res, retcode = self.run_oe_posit_docking(
                            opts_fred, pose_res, dus, lig_oemol, self.num_poses
                        )

                    if retcode == oedocking.OEDockingReturnCode_Success:
                        input_pairs = []
                        posed_ligands = []
                        num_poses = pose_res.GetNumPoses()
                        for i, result in enumerate(pose_res.GetSinglePoseResults()):
                            posed_mol = result.GetPose()
                            prob = result.GetProbability()

                            posed_ligand = Ligand.from_oemol(
                                posed_mol, **set.ligand.dict()
                            )
                            # set SD tags
                            sd_data = {
                                DockingResultCols.DOCKING_CONFIDENCE_POSIT.value: prob,
                                DockingResultCols.POSIT_METHOD.value: oedocking.OEPositMethodGetName(
                                    result.GetPositMethod()
                                ),
                                "Pose_ID": i,
                                "Num_Poses": num_poses,
                            }
                            posed_ligand.set_SD_data(sd_data)
                            posed_ligands.append(posed_ligand)

                            # Generate info about which target was actually used by multi-reference docking
                            if isinstance(set, DockingInputMultiStructure):
                                docked_target = set.complexes[result.GetReceptorIndex()]
                                input_pairs.append(
                                    DockingInputPair(
                                        ligand=set.ligand, complex=docked_target
                                    )
                                )
                            else:
                                input_pairs.append(set)

                        # Create Docking Results Objects
                        docking_results_objects = []
                        for input_pair, posed_ligand in zip(input_pairs, posed_ligands):
                            docking_results_objects.append(
                                POSITDockingResults(
                                    input_pair=input_pair,
                                    posed_ligand=posed_ligand,
                                    probability=posed_ligand.tags[
                                        DockingResultCols.DOCKING_CONFIDENCE_POSIT.value
                                    ],
                                    provenance=self.provenance(),
                                    pose_id=posed_ligand.tags["Pose_ID"],
                                    num_poses=posed_ligand.tags["Num_Poses"],
                                )
                            )
                        # Now we can decide if we want to return a path to the json file or the actual object
                        for docking_result in docking_results_objects:

                            if output_dir is not None:
                                json_path = docking_result.write_docking_files(
                                    output_dir
                                )
                            if return_for_disk_backend:
                                docking_results.append(json_path)
                            else:
                                docking_results.append(docking_result)

            except Exception as e:
                error_msg = f"docking failed for input pair with compound name: {set.ligand.compound_name}, smiles: {set.ligand.smiles} and target name: {set.complex.target.target_name} with error: {e}"
                if failure_mode == "skip":
                    logger.error(error_msg)
                elif failure_mode == "raise":
                    raise ValueError(error_msg)
                else:
                    raise ValueError(f"Unknown error handling option {failure_mode}")

        return docking_results

    def provenance(self) -> dict[str, str]:
        return {
            "oechem": oechem.OEChemGetVersion(),
            "oeomega": oeomega.OEOmegaGetVersion(),
            "oedocking": oedocking.OEDockingGetVersion(),
        }
