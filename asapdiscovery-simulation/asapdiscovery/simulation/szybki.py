import logging
import namedtuple
import pandas as pd

from asapdiscovery.data.openeye import oechem, oeszybki
from asapdiscovery.utils.logging import FileLogger

from openmm import unit
from pydantic import BaseModel
from pathlib import Path
from typing import Optional


class SzybkiFreeformResult(BaseModel):
    """
    Class for storing the results of a Szybki Freeform conformer analysis run.
    """

    ligand_name: str
    GlobalStrain: float
    LocalStrain: float
    ConformerStrain: float
    units = unit.kilocalories_per_mole

    class Config:
        allow_mutation = False


class SzybkiFreeformConformerAnalyzer:
    """
    Class for running ligand Szybki Freeform conformer analysis.
    """

    def __init__(
        self,
        ligand_paths: list[Path],
        output_paths: Optional[list[Path]] = None,
        logger: Optional[FileLogger] = None,
        debug: bool = False,
    ):
        """

        Parameters
        ----------
        ligand_paths : list[Path]
        output_paths : list[Path]
            List of paths to write the output to.
        logger : FileLogger
            Logger to use.
        debug : bool
            Whether to run in debug mode.
        """
        self.ligand_paths = ligand_paths

        if output_paths is None:
            outdir = Path("szybki").mkdir(exist_ok=True)
            self.output_paths = [outdir / ligand.parent for ligand in ligand_paths]
        else:
            self.output_paths = output_paths

        # init logger
        if logger is None:
            self.logger = FileLogger(
                "szybki_freeform_log.txt", "./", stdout=True, level=logging.INFO
            ).getLogger()
        else:
            self.logger = logger

        self.logger.info("Starting Szybki FreeForm conformer analysis run")
        self.debug = debug
        if self.debug:
            self.logger.SetLevel(logging.DEBUG)
            self.logger.debug("Running in debug mode")
        self.logger.debug(
            f"Running Szybki FreeForm on {len(self.ligand_paths)} ligands"
        )
        self.logger.debug(f"Writing to {self.output_paths}")

    def run_all_szybki(self, return_as_dataframe: bool = False):
        """
        Run Szybki on the ligands.
        """
        results = []
        for ligand_path, output_path in zip(self.ligand_paths, self.output_paths):
            self.logger.info(f"Running Szybki on {ligand_path}")
            self.logger.debug(f"Writing to {output_path}")
            results.append(self.run_szybki_on_ligand(ligand_path, output_path))
            self.logger.info(f"Finished Szybki on {ligand_path}")
        if return_as_dataframe:
            return pd.DataFrame(results)
        else:
            return results

    def run_szybki_on_ligand(self, ligand_path: Path, output_path: Path):
        """
        Run Szybki FreeForm on a single ligand.

        Adapted from the OpenEye example code for calculating advanced restriction energies (listing 20)
        https://docs.eyesopen.com/toolkits/python/szybkitk/examples.html

        Parameters
        ----------
        ligand_path : Path
            Path to the ligand.
        output_path : Path
            Path to write the output to.
        """

        if not ligand_path.exists():
            raise FileNotFoundError(f"{ligand_path} does not exist")

        if not output_path.exists():
            output_path.mkdir(exist_ok=True)

        # grab ligand name
        ligand_name = ligand_path.stem

        ensemble_output_sdf = output_path / "szybki_ensemble.sdf"

        # set up logging
        errfs = oechem.oeofstream("openeye_szybki_ensemble.txt")
        oechem.OEThrow.SetOutputStream(errfs)
        oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Debug)
        oechem.OEThrow.Info(f"Starting szybki freeform for {ligand_path}")

        # use a FileLogger with no formatting to write the info we want to capture from calculation to a file
        szybki_logger = FileLogger(
            str(output_path / "szybki_ensemble_log.txt"),
            "./",
            stdout=False,
            format="%(message)s",
            level=logging.INFO,
        ).getLogger()

        ifs = oechem.oemolistream()
        if not ifs.open(str(ligand_path)):
            oechem.OEThrow.Fatal(f"Unable to open {ligand_path} for reading")

        ofs = oechem.oemolostream()
        if not ofs.open(str(ensemble_output_sdf)):
            oechem.OEThrow.Fatal(f"Unable to open {ensemble_output_sdf} for writing")

        mol = oechem.OEMol()
        oechem.OEReadMolecule(ifs, mol)

        opts = oeszybki.OEFreeFormConfOptions()
        ffconf = oeszybki.OEFreeFormConfAdvanced(opts)

        # Make a copy of our MCMol.  We will execute the FreeFormConf commands on
        # the copied molecule so that our original molecule stays intact.
        omol = oechem.OEMol(mol)

        # Make further copies of our original molecule.  The copied molecule(s) would be used
        # as source on which retriction energies would be calculated
        rmol = oechem.OEMol(mol)
        fmol = oechem.OEMol(mol)

        # Prepare a comprehensive ensemble of molecule conformers.  For calculation
        # of restriction energies we want to make sure that all the corresponding free
        # conformers are also part of the comprehensive ensemble.  This will also
        # assign solvent charges on the molecule and check that the ensemble is
        # otherwise ready for FreeFormConf calculations. The resulting `fmol`
        # contains the correspondig free conformers.
        if not (
            ffconf.PrepareEnsemble(omol, rmol, fmol)
            == oeszybki.OEFreeFormReturnCode_Success
        ):
            oechem.OEThrow.Error(
                "Failed to prepare ensemble for FreeFormConf calculations"
            )

        # Perform loose optimization of the ensemble conformers.  We will remove
        # duplicates based on the loose optimization, to reduce the time needed for
        # tighter, more stricter optimization
        if not (
            ffconf.PreOptimizeEnsemble(omol) == oeszybki.OEFreeFormReturnCode_Success
        ):
            oechem.OEThrow.Error("Pre-optimization of the ensembles failed")

        # Remove duplicates from the pre-optimized ensemble
        if not (ffconf.RemoveDuplicates(omol) == oeszybki.OEFreeFormReturnCode_Success):
            oechem.OEThrow.Error("Duplicate removal from the ensembles failed")

        # Perform the desired optimization.  This uses a stricter convergence
        # criteria in the default settings.
        if not (ffconf.Optimize(omol) == oeszybki.OEFreeFormReturnCode_Success):
            oechem.OEThrow.Error("Optimization of the ensembles failed")

        # Remove duplicates to obtain the set of minimum energy conformers
        if not (ffconf.RemoveDuplicates(omol) == oeszybki.OEFreeFormReturnCode_Success):
            oechem.OEThrow.Error("Duplicate removal from the ensembles failed")

        # Perform FreeFormConf free energy calculations.  When all the above steps
        # have already been performed on the ensemble, this energy calculation
        # step is fast.
        if not (ffconf.EstimateEnergies(omol) == oeszybki.OEFreeFormReturnCode_Success):
            oechem.OEThrow.Error("Estimation of FreeFormConf energies failed")

        # Gather results of calculation into a results object for ease of viewing, etc.
        # then write to the log file
        res = oeszybki.OEFreeFormConfResults(omol)
        szybki_logger.info(f"Number of unique conformations: {res.GetNumUniqueConfs()}")
        szybki_logger.info("Conf.  Delta_G   Vibrational_Entropy")
        szybki_logger.info("      [kcal/mol]     [J/(mol K)]")
        for r in res.GetResultsForConformations():
            szybki_logger.info(
                f"{r.GetConfIdx():2d} {r.GetDeltaG():10.2f} {r.GetVibrationalEntropy():14.2f}"
            )

        # Identify the corresponding conformer(s) to the free minimized conformer(s).
        # If identified, the corresponding (Conf)Free energy information is also
        # copied to the free conformers
        if not (
            ffconf.IdentifyConformer(fmol, omol)
            == oeszybki.OEFreeFormReturnCode_Success
        ):
            oechem.OEThrow.Error("Identification of free conformer(s) failed")

        # Estimate restriction energies. Since both restricted and free conformer
        # energy components are already available, this operation is fast.
        if not (
            ffconf.EstimateRestrictionEnergy(fmol, rmol)
            == oeszybki.OEFreeFormReturnCode_Success
        ):
            oechem.OEThrow.Error("Restriction energy estimation failed")

        # Gather restriction energies into a results object for ease of viewing, etc.
        rstrRes = oeszybki.OERestrictionEnergyResult(fmol)
        szybki_logger.info(f"Unoptimised Global strain: {rstrRes.GetGlobalStrain()}")
        szybki_logger.info(f"Unoptimised Local strain:{rstrRes.GetLocalStrain()}")

        # It is much better to perform a restrained optimization of the
        # restricted conformer(s) to brush out any energy differences due to
        # force field constaints or the sources of conformer coordinates.  Note: The
        # high level EstimateFreeEnergy method does not perform this opertion.
        if not (
            ffconf.OptimizeRestraint(rmol) == oeszybki.OEFreeFormReturnCode_Success
        ):
            oechem.OEThrow.Error("Restraint optimization of the conformer(s) failed")

        # Estimate restriction energies on this optimized conformers.
        # Since both restricted and free conformer energy components
        # are already available, this operation is fast.
        if not (
            ffconf.EstimateRestrictionEnergy(fmol, rmol)
            == oeszybki.OEFreeFormReturnCode_Success
        ):
            oechem.OEThrow.Error("Restriction energy estimation failed")

        # Gather restriction energies into a results object for ease of viewing, etc.
        rstrRes = oeszybki.OERestrictionEnergyResult(fmol)
        szybki_logger.info(f"Optimised Global strain: {rstrRes.GetGlobalStrain()}")
        szybki_logger.info(f"Optimised Local strain:{rstrRes.GetLocalStrain()}")

        oechem.OEWriteMolecule(ofs, omol)

        # flush and close openeye log streams
        errfs.flush()
        errfs.close()

        # build the SzybkiFreeformResult object
        res = SzybkiFreeformResult(
            name=ligand_name,
            GlobalStrain=rstrRes.GetGlobalStrain(),
            LocalStrain=rstrRes.GetLocalStrain(),
            ConformerStrain=rstrRes.GetGlobalStrain() - rstrRes.GetLocalStrain(),
        )

        return res
