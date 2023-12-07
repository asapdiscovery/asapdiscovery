"""
This module contains the inputs, docker, and output schema for using POSIT
"""
import logging
from enum import Enum
from pathlib import Path
from typing import Literal, Optional, Union

from asapdiscovery.data.openeye import oechem, oedocking, oeomega
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.docking.docking_data_validation import (
    DockingResultColsV2 as DockingResultCols,
)
from asapdiscovery.docking.docking_v2 import (
    DockingBase,
    DockingInputMultiStructure,
    DockingInputPair,
    DockingResult,
)
from pydantic import Field, PositiveInt, root_validator

logger = logging.getLogger(__name__)


class POSIT_METHOD(Enum):
    """
    Enum for POSIT methods
    """

    ALL = oedocking.OEPositMethod_ALL
    HYBRID = oedocking.OEPositMethod_HYBRID
    FRED = oedocking.OEPositMethod_FRED
    MCS = oedocking.OEPositMethod_MCS
    SHAPEFIT = oedocking.OEPositMethod_SHAPEFIT

    @classmethod
    def reverse_lookup(cls, value):
        return cls(value).name


class POSIT_RELAX_MODE(Enum):
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
            docking_dict[
                DockingResultCols.LIGAND_ID
            ] = result.input_pair.ligand.compound_name
            docking_dict[
                DockingResultCols.TARGET_ID
            ] = result.input_pair.complex.target.target_name
            docking_dict[
                "target_bound_compound_smiles"
            ] = result.input_pair.complex.ligand.smiles
            docking_dict[DockingResultCols.SMILES] = result.input_pair.ligand.smiles
            docking_dict[
                DockingResultCols.DOCKING_CONFIDENCE_POSIT
            ] = result.probability
            docking_dict[DockingResultCols.DOCKING_SCORE_POSIT] = result.score
            docking_dict["score_type"] = result.score_type.value
            df_prep.append(docking_dict)

        df = pd.DataFrame(df_prep)
        return df


class POSITDocker(DockingBase):
    type: Literal["POSITDocker"] = "POSITDocker"

    relax: POSIT_RELAX_MODE = Field(
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
    allow_low_posit_prob: bool = Field(False, description="Allow low posit probability")
    low_posit_prob_thresh: float = Field(
        0.1,
        description="Minimum posit probability threshold if allow_low_posit_prob is False",
    )
    allow_final_clash: bool = Field(
        False, description="Allow clashing poses in last stage of docking"
    )
    allow_retries: bool = Field(
        True,
        description="Allow retries with different options if docking fails initially",
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

    def _dock(
        self,
        inputs: list[
            Union[
                DockingInputPair,
                DockingInputMultiStructure,
            ]
        ],
        output_dir: Optional[Union[str, Path]] = None,
        error="skip",
    ) -> list[DockingResult]:
        """
        Docking workflow using OEPosit
        """

        docking_results = []

        for set in inputs:
            if (
                set.is_cacheable
                and (output_dir is not None)
                and (
                    Path(
                        Path(output_dir) / set.unique_name() / "docking_result.json"
                    ).exists()
                )
            ):
                print(
                    f"Docking result for {set.unique_name()} already exists, reading from disk"
                )
                output_dir = Path(output_dir)
                docking_results.append(
                    POSITDockingResults.from_json_file(
                        output_dir / set.unique_name() / "docking_result.json"
                    )
                )
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
                    omega = oeomega.OEOmega(omegaOpts)
                    omega_retcode = omega.Build(lig_oemol)
                    if omega_retcode:
                        error_msg = f"Omega failed with error code: {omega_retcode} : {oeomega.OEGetOmegaError(omega_retcode)}"
                        if error == "skip":
                            logger.error(error_msg)
                        elif error == "raise":
                            raise ValueError(error_msg)
                        else:
                            raise ValueError(f"Unknown error handling option {error}")

                opts = oedocking.OEPositOptions()
                opts.SetIgnoreNitrogenStereo(True)
                opts.SetPositMethods(self.posit_method.value)
                opts.SetPoseRelaxMode(self.relax.value)

                pose_res = oedocking.OEPositResults()
                pose_res, retcode = self.run_oe_posit_docking(
                    opts, pose_res, dus, lig_oemol, self.num_poses
                )

                if self.allow_retries:
                    # try again with no relaxation
                    if retcode == oedocking.OEDockingReturnCode_NoValidNonClashPoses:
                        opts.SetPoseRelaxMode(oedocking.OEPoseRelaxMode_NONE)
                        pose_res, retcode = self.run_oe_posit_docking(
                            opts, pose_res, dus, lig_oemol, self.num_poses
                        )

                    # try again with low posit probability
                    if (
                        retcode == oedocking.OEDockingReturnCode_NoValidNonClashPoses
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

                if retcode == oedocking.OEDockingReturnCode_Success:
                    for result in pose_res.GetSinglePoseResults():
                        posed_mol = result.GetPose()
                        prob = result.GetProbability()

                        posed_ligand = Ligand.from_oemol(posed_mol, **set.ligand.dict())
                        # set SD tags
                        sd_data = {
                            DockingResultCols.DOCKING_CONFIDENCE_POSIT.value: prob,
                            DockingResultCols.POSIT_METHOD.value: POSIT_METHOD.reverse_lookup(
                                self.posit_method.value
                            ),
                        }
                        posed_ligand.set_SD_data(sd_data)

                        # Generate info about which target was actually used by multi-reference docking
                        if isinstance(set, DockingInputMultiStructure):
                            docked_target = set.complexes[result.GetReceptorIndex()]
                            input_pair = DockingInputPair(
                                ligand=set.ligand, complex=docked_target
                            )
                        else:
                            input_pair = set

                        docking_result = POSITDockingResults(
                            input_pair=input_pair,
                            posed_ligand=posed_ligand,
                            probability=prob,
                            provenance=self.provenance(),
                        )
                        docking_results.append(docking_result)
                        if output_dir is not None:
                            docking_result.write_docking_files(output_dir)

                else:
                    if error == "skip":
                        logger.warn(
                            f"docking failed for input pair with compound name: {set.ligand.compound_name}, smiles: {set.ligand.smiles} and target name: {set.complex.target.target_name}"
                        )
                        docking_results.append(None)
                    elif error == "raise":
                        raise ValueError(
                            f"docking failed for input pair with compound name: {set.ligand.compound_name}, smiles: {set.ligand.smiles} and target name: {set.complex.target.target_name}"
                        )
                    else:
                        raise ValueError(f"Unknown error handling option {error}")

        return docking_results

    def provenance(self) -> dict[str, str]:
        return {
            "oechem": oechem.OEChemGetVersion(),
            "oeomega": oeomega.OEOmegaGetVersion(),
            "oedocking": oedocking.OEDockingGetVersion(),
        }
