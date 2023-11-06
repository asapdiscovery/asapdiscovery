import abc
from enum import Enum
from pathlib import Path
from typing import Literal, Optional, Union

import dask
from asapdiscovery.data.dask_utils import actualise_dask_delayed_iterable
from asapdiscovery.data.openeye import (
    combine_protein_ligand,
    oechem,
    oedocking,
    oeomega,
    save_openeye_pdb,
)
from asapdiscovery.data.schema_v2.ligand import Ligand, compound_names_unique
from asapdiscovery.data.schema_v2.pairs import DockingInputPair
from asapdiscovery.docking.docking_data_validation import (
    DockingResultColsV2 as DockingResultCols,
)
from asapdiscovery.modeling.modeling import split_openeye_design_unit
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, root_validator


class DockingResult(BaseModel):
    """
    Schema for a DockingResult, containing both a DockingInputPair used as input to the workflow
    and a Ligand object containing the docked pose.
    Also contains the probability of the docked pose if applicable.

    Parameters
    ----------
    input_pair : DockingInputPair
        Input pair
    posed_ligand : Ligand
        Posed ligand
    probability : float, optional
        Probability of the docked pose, by default None
    provenance : dict[str, str]
        Provenance information

    """

    type: Literal["DockingResult"] = "DockingResult"
    input_pair: DockingInputPair = Field(description="Input pair")
    posed_ligand: Ligand = Field(description="Posed ligand")
    probability: Optional[PositiveFloat] = Field(
        description="Probability"
    )  # not easy to get the probability from rescoring
    provenance: dict[str, str] = Field(description="Provenance")

    def get_output(self) -> dict:
        """
        return a dictionary of some of the fields of the DockingResult
        """
        dct = self.dict()
        dct.pop("input_pair")
        dct.pop("posed_ligand")
        dct.pop("type")
        return dct

    def to_posed_oemol(self) -> oechem.OEMol:
        """
        Combine the original target and posed ligand into a single oemol

        Returns
        -------
        oechem.OEMol
            Combined oemol
        """
        return combine_protein_ligand(self.to_protein(), self.posed_ligand.to_oemol())
    
    def to_protein(self) -> oechem.OEMol:
        """
        Return the protein from the original target

        Returns
        -------
        oechem.OEMol
            Protein oemol
        """
        _, prot, _ = split_openeye_design_unit(self.input_pair.complex.target.to_oedu())
        return prot

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
            Dataframe of DockingResults
        """
        import pandas as pd

        return pd.DataFrame([r.get_output() for r in results])


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


class DockingBase(BaseModel):
    """
    Base class for running docking
    """

    type: Literal["DockingBase"] = "DockingBase"

    @abc.abstractmethod
    def _dock() -> list[DockingResult]:
        ...

    def dock(
        self, inputs: list[DockingInputPair], use_dask: bool = False, dask_client=None
    ) -> Union[list[dask.delayed], list[DockingResult]]:
        if use_dask:
            delayed_outputs = []
            for inp in inputs:
                out = dask.delayed(self._dock)(inputs=[inp])
                delayed_outputs.append(out[0])  # flatten
            outputs = actualise_dask_delayed_iterable(
                delayed_outputs, dask_client=dask_client, errors="skip"
            )
        else:
            outputs = self._dock(inputs=inputs)
        # filter out None values
        outputs = [o for o in outputs if o is not None]
        return outputs

    @abc.abstractmethod
    def provenance(self) -> dict[str, str]:
        ...


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


class POSITDocker(DockingBase):
    """
    Docking workflow using OEPosit

    Parameters
    ----------
    relax : POSIT_RELAX_MODE
        whether to allow receptor relaxation either, 'clash', 'all', 'none', by default POSIT_RELAX_MODE.NONE
    posit_method : POSIT_METHOD
        POSIT method to use, by default POSIT_METHOD.ALL
    use_omega : bool
        Whether to use OEOmega to generate conformers, by default True
    num_poses : PositiveInt
        Number of poses to generate, by default 1
    allow_low_posit_prob : bool
        Whether to allow low posit probability, by default False
    low_posit_prob_thresh : float
        Minimum posit probability threshold if allow_low_posit_prob is False, by default 0.1
    allow_final_clash : bool
        Whether to allow clashing poses in last stage of docking, by default False
    """

    type: Literal["POSITDocker"] = "POSITDocker"

    relax: POSIT_RELAX_MODE = Field(
        POSIT_RELAX_MODE.NONE,
        description="When to check for relaxation either, 'clash', 'all', 'none'",
    )
    posit_method: POSIT_METHOD = Field(
        POSIT_METHOD.ALL, description="POSIT method to use"
    )
    use_omega: bool = Field(True, description="Use omega to generate conformers")
    num_poses: PositiveInt = Field(1, description="Number of poses to generate")
    allow_low_posit_prob: bool = Field(False, description="Allow low posit probability")
    low_posit_prob_thresh: float = Field(
        0.1,
        description="Minimum posit probability threshold if allow_low_posit_prob is False",
    )
    allow_final_clash: bool = Field(
        False, description="Allow clashing poses in last stage of docking"
    )

    @staticmethod
    def to_result_type():
        return POSITDockingResults

    @root_validator
    @classmethod
    def _output_dir_write_file(cls, values):
        output_dir = values.get("output_dir")
        write_files = values.get("write_file")
        if write_files and not output_dir:
            raise ValueError("Output directory must be set if write_file is True")

        if write_files and not Path(output_dir).exists():
            raise ValueError("Output directory does not exist")
        return values

    @staticmethod
    def run_oe_posit_docking(opts, pose_res, du, lig, num_poses):
        """
        Helper function to run OEPosit docking

        Parameters
        ----------
        opts : oedocking.OEPositOptions
            OEPosit options
        pose_res : oedocking.OEPositResults
            OEPosit results
        du : oedocking.OEDesignUnit
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
        retcode = poser.AddReceptor(du)
        if not retcode:
            raise ValueError("Failed to add receptor to POSIT")
        ret_code = poser.Dock(pose_res, lig, num_poses)
        return pose_res, ret_code

    def _dock(
        self, inputs: list[DockingInputPair], error="skip"
    ) -> list[DockingResult]:
        """
        Docking workflow using OEPosit
        """

        docking_results = []

        for pair in inputs:
            du = pair.complex.target.to_oedu()
            lig_oemol = oechem.OEMol(pair.ligand.to_oemol())
            if self.use_omega:
                omegaOpts = oeomega.OEOmegaOptions()
                omega = oeomega.OEOmega(omegaOpts)
                ret_code = omega.Build(lig_oemol)
                if ret_code:
                    if error == "skip":
                        print(
                            f"Omega failed with error code {oeomega.OEGetOmegaError(ret_code)}"
                        )
                    elif error == "raise":
                        raise ValueError(
                            f"Omega failed with error code {oeomega.OEGetOmegaError(ret_code)}"
                        )
                    else:
                        raise ValueError(f"Unknown error handling option {error}")

            opts = oedocking.OEPositOptions()
            opts.SetIgnoreNitrogenStereo(True)
            opts.SetPositMethods(self.posit_method.value)
            opts.SetPoseRelaxMode(self.relax.value)

            pose_res = oedocking.OEPositResults()
            pose_res, retcode = self.run_oe_posit_docking(
                opts, pose_res, du, lig_oemol, self.num_poses
            )

            # TODO: all this retrying is very inefficient, this should be able to be done faster.

            # try again with no relaxation
            if retcode == oedocking.OEDockingReturnCode_NoValidNonClashPoses:
                opts.SetPoseRelaxMode(oedocking.OEPoseRelaxMode_NONE)
                pose_res, retcode = self.run_oe_posit_docking(
                    opts, pose_res, du, lig_oemol, self.num_poses
                )

            # try again with low posit probability
            if (
                retcode == oedocking.OEDockingReturnCode_NoValidNonClashPoses
                and self.allow_low_posit_prob
            ):
                opts.SetPoseRelaxMode(oedocking.OEPoseRelaxMode_ALL)
                opts.SetMinProbability(self.low_posit_prob_thresh)
                pose_res, retcode = self.run_oe_posit_docking(
                    opts, pose_res, du, lig_oemol, self.num_poses
                )

            # try again allowing clashes
            if (
                self.allow_final_clash
                and ret_code == oedocking.OEDockingReturnCode_NoValidNonClashPoses
            ):
                opts.SetPoseRelaxMode(oedocking.OEPoseRelaxMode_ALL)
                opts.SetAllowedClashType(oedocking.OEAllowedClashType_ANY)
                pose_res, retcode = self.run_oe_posit_docking(
                    opts, pose_res, du, lig_oemol, self.num_poses
                )

            if ret_code == oedocking.OEDockingReturnCode_Success:
                for result in pose_res.GetSinglePoseResults():
                    posed_mol = result.GetPose()
                    prob = result.GetProbability()
                    pose_scorer = oedocking.OEScore(oedocking.OEScoreType_Chemgauss4)
                    pose_scorer.Initialize(du)
                    chemgauss_score = pose_scorer.ScoreLigand(posed_mol)

                    posed_ligand = Ligand.from_oemol(posed_mol, **pair.ligand.dict())
                    # set SD tags
                    sd_data = {
                        DockingResultCols.DOCKING_CONFIDENCE_POSIT.value: prob,
                        DockingResultCols.DOCKING_SCORE_POSIT.value: chemgauss_score,
                        DockingResultCols.POSIT_METHOD.value: POSIT_METHOD.reverse_lookup(
                            self.posit_method.value
                        ),
                    }
                    posed_ligand.set_SD_data(sd_data)

                    docking_result = POSITDockingResults(
                        input_pair=pair,
                        posed_ligand=posed_ligand,
                        probability=prob,
                        provenance=self.provenance(),
                    )
                    docking_results.append(docking_result)

            else:
                if error == "skip":
                    print(
                        f"docking failed for input pair with compound name: {pair.ligand.compound_name}, smiles: {pair.ligand.smiles} and target name: {pair.complex.target.target_name}"
                    )
                    docking_results.append(None)
                elif error == "raise":
                    raise ValueError(
                        f"docking failed for input pair with compound name: {pair.ligand.compound_name}, smiles: {pair.ligand.smiles} and target name: {pair.complex.target.target_name}"
                    )
                else:
                    raise ValueError(f"Unknown error handling option {error}")

        return docking_results

    @staticmethod
    def write_docking_files(
        docking_results: list[DockingResult], output_dir: Union[str, Path]
    ):
        """
        Write docking results to files in output_dir, directories will have the form:
        {target_name}_+_{ligand_name}/docked.sdf
        {target_name}_+_{ligand_name}/docked_complex.pdb

        Parameters
        ----------
        docking_results : list[DockingResult]
            List of DockingResults
        output_dir : Union[str, Path]
            Output directory

        Raises
        ------
        ValueError
            If compound names of input pair and posed ligand do not match

        """
        ligs = [docking_result.input_pair.ligand for docking_result in docking_results]
        names_unique = compound_names_unique(ligs)
        output_dir = Path(output_dir)
        # if names are not unique, we will use unknown_ligand_{i} as the ligand portion of directory
        # when writing files

        # write out the docked pose
        for i, result in enumerate(docking_results):
            if (
                not result.input_pair.ligand.compound_name
                == result.posed_ligand.compound_name
            ):
                raise ValueError(
                    "Compound names of input pair and posed ligand do not match"
                )
            if names_unique:
                output_pref = (
                    result.input_pair.complex.target.target_name
                    + "_+_"
                    + result.posed_ligand.compound_name
                )
            else:
                output_pref = (
                    result.input_pair.complex.target.target_name
                    + "_+_"
                    + f"unknown_ligand_{i}"
                )

            compound_dir = output_dir / output_pref
            compound_dir.mkdir(parents=True, exist_ok=True)
            output_sdf_file = compound_dir / "docked.sdf"
            output_pdb_file = compound_dir / "docked_complex.pdb"

            result.posed_ligand.to_sdf(output_sdf_file)

            combined_oemol = result.to_posed_oemol()
            save_openeye_pdb(combined_oemol, output_pdb_file)

    def provenance(self) -> dict[str, str]:
        return {
            "oechem": oechem.OEChemGetVersion(),
            "oeomega": oeomega.OEOmegaGetVersion(),
            "oedocking": oedocking.OEDockingGetVersion(),
        }
