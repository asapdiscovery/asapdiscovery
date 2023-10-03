from pydantic import BaseModel, Field, PositiveInt, root_validator
from pathlib import Path
from asapdiscovery.data.openeye import oedocking, oeomega, oechem, save_openeye_pdb
from asapdiscovery.data.schema_v2.pairs import DockingInputPair
from asapdiscovery.data.schema_v2.ligand import Ligand, compound_names_unique
from asapdiscovery.docking.docking_data_validation import DockingResultCols
from enum import Enum
import pandas as pd
import abc

class DockingResult(BaseModel):
    input_pair: DockingInputPair = Field(description="Input pair")
    posed_ligand: Ligand = Field(description="Posed ligand")
    probability: float = Field(description="Probability")
    chemgauss_score: float = Field(description="Chemgauss4 score")
    provenance: dict[str, str] = Field(description="Provenance")

    @root_validator
    @classmethod
    def smiles_match(cls, values):
        posed_ligand = values.get("posed_ligand")
        input_pair = values.get("input_pair")
        if posed_ligand.smiles != input_pair.ligand.smiles:
            raise ValueError(
                "SMILES of ligand and ligand in input docking pair not match"
            )
        return values

class DockingBase(BaseModel):
    """
    Base class for docking.
    """

    @abc.abstractmethod
    def _dock() -> list[DockingResult]:
        ...

    def dock(self, inputs: list[DockingInputPair]) -> list[DockingResult]:
        outputs = self._dock(inputs=inputs)
        return outputs

    @abc.abstractmethod
    def provenance(self) -> dict[str, str]:
        ...


class POSIT_METHOD(Enum):
    ALL = oedocking.OEPositMethod_ALL
    HYBRID = oedocking.OEPositMethod_HYBRID
    FRED = oedocking.OEPositMethod_FRED
    MCS = oedocking.OEPositMethod_MCS
    SHAPEFIT = oedocking.OEPositMethod_SHAPEFIT


class POSIT_RELAX_MODE(Enum):
    CLASH = oedocking.OEPoseRelaxMode_CLASHED
    ALL = oedocking.OEPoseRelaxMode_ALL
    NONE = oedocking.OEPoseRelaxMode_NONE


class POSITDocker(DockingBase):
    """
    Docker class for POSIT.
    """

    relax: POSIT_RELAX_MODE = Field(
        POSIT_RELAX_MODE.NONE,
        description="When to check for relaxation either, 'clash', 'all', 'none'",
    )
    posit_method: POSIT_METHOD = Field(
        POSIT_METHOD.ALL, description="POSIT method to use"
    )
    use_omega: bool = Field(True, description="Use omega to generate conformers")
    num_poses: PositiveInt = Field(1, description="Number of poses to generate")
    log_name: str = Field("run_docking_oe", description="Name of the log file")
    openeye_logname: str = Field(
        "openeye-log.txt", description="Name of the openeye log file"
    )
    allow_low_posit_prob: bool = Field(False, description="Allow low posit probability")
    low_posit_prob_thresh: float = Field(
        0.1,
        description="Minimum posit probability threshold if allow_low_posit_prob is True",
    )
    allow_final_clash: bool = Field(
        False, description="Allow clashing poses in last stage of docking"
    )
    output_dir: str = Field(
        "docking", description="Output directory for docking results"
    )
    write_files: bool = Field(False, description="Write docked pose results to file")

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
        poser = oedocking.OEPosit(opts)
        retcode = poser.AddReceptor(du)
        if not retcode:
            raise ValueError("Failed to add receptor to POSIT")
        ret_code = poser.Dock(pose_res, lig, num_poses)
        return pose_res, ret_code

    def _dock(self, inputs: list[DockingInputPair]) -> list[DockingResult]:
        """
        Dock the inputs
        """

        ligs = [pair.ligand for pair in inputs]
        names_unique = compound_names_unique(ligs)
        # if names are not unique, we will use unknown_ligand_{i} as the output directory
        # when writing files

        docking_results = []

        for pair in inputs:
            du = pair.complex.target.to_oedu()
            lig = pair.ligand
            lig_oemol = oechem.OEMol(pair.ligand.to_oemol())
            if self.use_omega:
                omegaOpts = oeomega.OEOmegaOptions()
                omega = oeomega.OEOmega(omegaOpts)
                ret_code = omega.Build(lig_oemol)
                if ret_code:
                    raise ValueError(
                        f"Omega failed with error code {oeomega.OEGetOmegaError(ret_code)}"
                    )

            opts = oedocking.OEPositOptions()
            opts.SetIgnoreNitrogenStereo(True)
            opts.SetPositMethods(self.posit_method.value)
            opts.SetPoseRelaxMode(self.relax.value)

            pose_res = oedocking.OEPositResults()
            pose_res, retcode = self.run_oe_posit_docking(
                opts, pose_res, du, lig_oemol, self.num_poses
            )

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
                        DockingResultCols.POSIT_METHOD.value: self.posit_method.value,
                    }
                    posed_ligand.set_SD_data(sd_data)

                    docking_result = DockingResult(
                        input_pair=pair,
                        posed_ligand=posed_ligand,
                        probability=prob,
                        chemgauss_score=chemgauss_score,
                        provenance=self.provenance(),
                    )
                    docking_results.append(docking_result)

                    if self.write_files:
                        # write out the docked pose
                        if names_unique:
                            output_dir = self.output_dir / lig.compound_name
                        else:
                            output_dir = self.output_dir / f"unknown_ligand_{i}"
                        output_dir.mkdir(parents=True, exist_ok=True)
                        output_sdf_file = output_dir / "docked.sdf"
                        output_pdb_file = output_dir / "docked_complex.pdb"

                        posed_ligand.to_sdf_file(output_sdf_file)

                        combined_oemol = docking_result.to_posed_oemol()
                        save_openeye_pdb(output_pdb_file, combined_oemol)

            else:
                pass

        return docking_results

    def provenance(self) -> dict[str, str]:
        return {"oedocking": "POSIT"}
