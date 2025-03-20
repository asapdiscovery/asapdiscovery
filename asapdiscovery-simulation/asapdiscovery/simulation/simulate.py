import abc
import logging
import shutil
import warnings
from pathlib import Path
from typing import Any, ClassVar, Optional, Union  # noqa: F401

import mdtraj
import openmm
import pandas as pd
from asapdiscovery.data.backend.openeye import save_openeye_pdb
from asapdiscovery.data.util.dask_utils import (
    BackendType,
    FailureMode,
    backend_wrapper,
    dask_vmap,
)
from asapdiscovery.data.util.stringenum import StringEnum
from asapdiscovery.docking.docking import DockingResult
from mdtraj.core.residue_names import _SOLVENT_TYPES
from mdtraj.reporters import XTCReporter
from multimethod import multimethod
from openff.toolkit.topology import Molecule
from openmm import LangevinMiddleIntegrator, MonteCarloBarostat, Platform, app, unit
from openmm.app import Modeller, PDBFile, Simulation, StateDataReporter
from openmmforcefields.generators import SystemGenerator
from openmmtools.utils import get_fastest_platform
from pydantic.v1 import (
    BaseModel,
    Field,
    PositiveFloat,
    PositiveInt,
    root_validator,
    validator,
)
from rdkit import Chem
from tqdm import tqdm

logger = logging.getLogger(__name__)


solvent_types = list(_SOLVENT_TYPES)


class OpenMMPlatform(StringEnum):
    """
    Enum for OpenMM platforms.
    """

    CPU = "CPU"
    CUDA = "CUDA"
    OpenCL = "OpenCL"
    Reference = "Reference"
    Fastest = "Fastest"

    def get_platform(self):
        if Platform.getNumPlatforms() == 0:
            raise ValueError("No compatible OpenMM patforms detected")

        if self.value == "Fastest":
            return get_fastest_platform()
        else:
            return Platform.getPlatformByName(self.value)


def truncate_num_steps(num_steps, reporting_interval):
    # Ensure num_steps is at least one reporting interval
    num_steps = max(num_steps, reporting_interval)
    # Truncate num_steps to be a multiple of reporting_interval
    num_steps = (num_steps // reporting_interval) * reporting_interval

    return num_steps


class SimulatorBase(BaseModel):
    """
    Base class for Simulators.
    """

    output_dir: Path = Field("md", description="Output directory")
    debug: bool = Field(False, description="Debug mode of the simulation")

    @abc.abstractmethod
    def _simulate(self) -> list["SimulationResult"]: ...

    def simulate(
        self,
        inputs: list[DockingResult],
        use_dask: bool = False,
        dask_client=None,
        failure_mode=FailureMode.SKIP,
        backend=BackendType.IN_MEMORY,
        reconstruct_cls=None,
        **kwargs,
    ) -> pd.DataFrame:

        return self._simulate(
            inputs=inputs,
            use_dask=use_dask,
            dask_client=dask_client,
            failure_mode=failure_mode,
            backend=backend,
            reconstruct_cls=reconstruct_cls,
            **kwargs,
        )

    @abc.abstractmethod
    def provenance(self) -> dict[str, str]: ...


class SimulationResult(BaseModel):
    traj_path: Path
    minimized_pdb_path: Path
    final_pdb_path: Optional[Path]
    success: Optional[bool]
    input_docking_result: Optional[DockingResult]


class VanillaMDSimulator(SimulatorBase):
    collision_rate: PositiveFloat = Field(
        1, description="Collision rate of the simulation (in 1/ps)"
    )
    openmm_logname: str = Field(
        "openmm_log.tsv", description="Name of the OpenMM log file"
    )
    openmm_platform: OpenMMPlatform = Field(
        OpenMMPlatform.Fastest, description="OpenMM platform to use"
    )
    temperature: PositiveFloat = Field(
        300, description="Temperature of the simulation (in kelvin)"
    )
    pressure: PositiveFloat = Field(
        1, description="Pressure of the simulation (in atm)"
    )
    timestep: PositiveFloat = Field(
        4, description="Timestep of the simulation (in femtoseconds)"
    )
    equilibration_steps: PositiveInt = Field(
        5000, description="Number of equilibration steps"
    )
    reporting_interval: PositiveInt = Field(
        1250, description="Reporting interval of the simulation"
    )
    num_steps: PositiveInt = Field(
        2500000,
        description="Number of simulation steps, must be a multiple of reporting interval or will be truncated to nearest multiple of reporting interval",
    )

    progressbar: bool = Field(
        False, description="Whether to show a progress bar during simulation"
    )

    rmsd_restraint: bool = Field(
        False, description="Whether to apply an RMSD restraint to the simulation"
    )
    rmsd_restraint_atom_indices: list[int] = Field(
        [],
        description="Atom indices to apply the RMSD restraint to, cannot be used with rmsd_restraint_type",
    )
    rmsd_restraint_type: Optional[str] = Field(
        None,
        description="Type of RMSD restraint to apply, must be 'CA' or 'heavy', cannot be used with rmsd_restraint_atom_indices",
    )

    rmsd_restraint_force_constant: PositiveFloat = Field(
        50, description="Force constant of the RMSD restraint (in kcal/mol/A^2)"
    )

    truncate_steps: bool = Field(
        True,
        description="Whether to truncate num_steps to multiple of reporting interval, used mostly for testing",
    )

    small_molecule_force_field: str = Field(
        "openff-2.2.0",
        description="The OpenFF small molecule force field which should be used for the ligand.",
    )
    collect_dir: Optional[Path] = Field(
        None, description="Directory to collect results in a single directory"
    )
    minimize_only: bool = Field(
        False,
        description="Whether to carry out a single minimization step.",
    )

    @validator("rmsd_restraint_type")
    @classmethod
    def check_restraint_type(cls, v):
        if v not in ["CA", "heavy", None]:
            raise ValueError("RMSD restraint type must be 'CA' or 'heavy'")
        return v

    @validator("rmsd_restraint_atom_indices")
    @classmethod
    def check_restraint_atom_indices(cls, v):
        if len(v) == 0:
            return v
        if not isinstance(v, list):
            raise ValueError("RMSD restraint atom indices must be a list")
        if not all(isinstance(x, int) for x in v):
            raise ValueError("RMSD restraint atom indices must be a list of ints")
        return v

    @root_validator
    @classmethod
    def check_restraint_setup(cls, values):
        """
        Validate RMSD restraint setup
        """
        rmsd_restraint = values.get("rmsd_restraint")
        rmsd_restraint_indices = values.get("rmsd_restraint_indices")
        rmsd_restraint_type = values.get("rmsd_restraint_type")
        if rmsd_restraint_type and rmsd_restraint_indices:
            raise ValueError(
                "If RMSD restraint type is provided, rmsd_restraint_indices must be empty"
            )

        if (
            rmsd_restraint
            and not rmsd_restraint_type
            and len(rmsd_restraint_indices) == 0
        ):
            raise ValueError(
                "If RMSD restraint is enabled, and rmsd_restraint_type is not provided rmsd_restraint_indices must be provided"
            )
        return values

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    @root_validator
    @classmethod
    def check_and_apply_truncation(cls, values):
        """
        Validate num_steps and reporting_interval along with truncate_steps
        """
        step_truncation = values.get("truncate_steps")
        num_steps = values.get("num_steps")
        reporting_interval = values.get("reporting_interval")
        if step_truncation:
            values["num_steps"] = truncate_num_steps(num_steps, reporting_interval)
        return values

    @root_validator
    @classmethod
    def check_steps(cls, values):
        """
        Validate num_steps and reporting_interval
        """
        num_steps = values.get("num_steps")
        reporting_interval = values.get("reporting_interval")
        truncate_steps = values.get("truncate_steps")
        if (num_steps % reporting_interval != 0) and truncate_steps:
            raise ValueError(
                f"num_steps ({num_steps}) must be a multiple of reporting_interval ({reporting_interval})"
            )
        return values

    @property
    def n_frames(self) -> int:
        return self.num_steps // self.reporting_interval

    @property
    def total_simulation_time(self) -> openmm.unit.quantity.Quantity:
        return self.num_steps * self.timestep * unit.femtoseconds

    @property
    def frames_per_ns(self) -> unit.quantity.Quantity:
        # convert to ns
        length = (self.total_simulation_time).value_in_unit(unit.nanoseconds)
        return self.n_frames / length

    def _to_openmm_units(self) -> OpenMMPlatform:
        self._temperature = self.temperature * unit.kelvin
        self._pressure = self.pressure * unit.atmospheres
        self._collision_rate = self.collision_rate / unit.picoseconds
        self._timestep = self.timestep * unit.femtoseconds
        self.n_snapshots = int(self.num_steps / self.reporting_interval)
        self.num_steps = self.n_snapshots * self.reporting_interval
        # set platform
        _platform = OpenMMPlatform(self.openmm_platform).get_platform()
        if _platform.getName() == "CUDA" or _platform.getName() == "OpenCL":
            _platform.setPropertyDefaultValue("Precision", "mixed")

        if self.debug:
            _platform = OpenMMPlatform.CPU.get_platform()
        return _platform

    @dask_vmap(["inputs"], has_failure_mode=True)
    @backend_wrapper("inputs")
    def _simulate(
        self,
        inputs: Union[list[DockingResult], list[tuple[Path, Path]]],
        outpaths: Optional[list[Path]] = None,
        **kwargs,
    ) -> list[dict[str, str]]:
        if outpaths:
            if len(outpaths) != len(inputs):
                raise ValueError("outpaths must be the same length as inputs")

        return self._dispatch(inputs, outpaths=outpaths, **kwargs)

    @multimethod
    def _dispatch(
        self, inputs: list[DockingResult], failure_mode: str = "skip", **kwargs
    ):
        # outpaths is unused in this overload
        results = []
        for inp in inputs:
            try:
                output_pref = inp.unique_name
                outpath = self.output_dir / output_pref
                if not outpath.exists():
                    outpath.mkdir(parents=True)
                posed_sdf_path = outpath / "posed_ligand.sdf"
                inp.posed_ligand.to_sdf(posed_sdf_path)
                # write pdb to pre file
                pre_pdb_path = outpath / "pre.pdb"
                save_openeye_pdb(inp.to_protein(), pre_pdb_path)
                res = self._simulate_loop(
                    pre_pdb_path, posed_sdf_path, outpath, input_docking_result=inp
                )
                results.append(res)
            except Exception as e:
                if failure_mode == "skip":
                    logger.error(f"Error processing {inp.unique_name}: {e}")
                elif failure_mode == "raise":
                    raise e
                else:
                    raise ValueError(
                        f"Unknown error mode: {failure_mode}, must be 'skip' or 'raise'"
                    )
        return results

    @_dispatch.register
    def _dispatch(
        self,
        inputs: list[tuple[Path, Path]],
        outpaths: Optional[list[Path]] = None,
        failure_mode: str = "skip",
    ):
        results = []
        if not outpaths:
            outpaths = [None] * len(inputs)
        for (protein, ligand), outpath in zip(inputs, outpaths):
            try:
                tag = protein.stem + "_" + ligand.stem
                if outpath:
                    outpath = Path(outpath) / tag
                else:
                    outpath = self.output_dir / tag
                if not outpath.exists():
                    outpath.mkdir(parents=True)
                res = self._simulate_loop(protein, ligand, outpath)
                results.append(res)
            except Exception as e:
                if failure_mode == "skip":
                    logger.error(f"Error processing {tag}: {e}")
                elif failure_mode == "raise":
                    raise e
                else:
                    raise ValueError(
                        f"Unknown error mode: {failure_mode}, must be 'skip' or 'raise'"
                    )
        return results

    def _simulate_loop(
        self,
        protein: Path,
        ligand: Path,
        outpath: Path,
        input_docking_result: Optional[DockingResult] = None,
    ) -> list[SimulationResult]:
        logger.info(f"Running simulation for {protein.stem} and {ligand.stem}")
        _platform = self._to_openmm_units()
        processed_ligand = self.process_ligand_rdkit(ligand)
        system_generator, ligand_mol = self.create_system_generator(processed_ligand)
        logger.debug("Created system generator")
        modeller, ligand_mol = self.get_complex_model(ligand_mol, protein)

        modeller, mol_atom_indices = self.setup_and_solvate(
            system_generator, modeller, ligand_mol
        )
        logger.debug("Setup and solvated system")
        system, output_indices, output_topology = self.create_system(
            system_generator, modeller, mol_atom_indices, processed_ligand
        )
        logger.debug("Created system")
        simulation, context = self.setup_simulation(
            modeller, system, output_indices, output_topology, outpath, _platform
        )
        if self.minimize_only:
            sim_result = SimulationResult(
                input_docking_result=input_docking_result,
                traj_path="",
                minimized_pdb_path=outpath / "minimized.pdb",
                final_pdb_path="",
                success=True,
            )
            return sim_result

        logger.info("Setup simulation")
        simulation = self.equilibrate(simulation)
        logger.info("Equilibrated")
        logger.info("Running production simulation")
        retcode = self.run_production_simulation(
            simulation, context, output_indices, output_topology, outpath
        )
        logger.debug("Finished production simulation")

        sim_result = SimulationResult(
            input_docking_result=input_docking_result,
            traj_path=outpath / "traj.xtc",
            minimized_pdb_path=outpath / "minimized.pdb",
            final_pdb_path=outpath / "final.pdb",
            success=retcode,
        )

        if self.collect_dir:
            if input_docking_result:
                tag = input_docking_result.unique_name
            else:
                tag = protein.stem + "_" + ligand.stem

            if not self.collect_dir.exists():
                self.collect_dir.mkdir(parents=True)
            if sim_result.traj_path.exists():
                shutil.copy(sim_result.traj_path, self.collect_dir / f"{tag}_traj.xtc")
            if sim_result.minimized_pdb_path.exists():
                shutil.copy(
                    sim_result.minimized_pdb_path,
                    self.collect_dir / f"{tag}_minimized.pdb",
                )
            if sim_result.final_pdb_path.exists():
                shutil.copy(
                    sim_result.final_pdb_path, self.collect_dir / f"{tag}_final.pdb"
                )

        return sim_result

    @staticmethod
    def process_ligand_rdkit(sdf_path) -> Molecule:
        rdkitmol = Chem.SDMolSupplier(str(sdf_path))[0]
        rdkitmolh = Chem.AddHs(rdkitmol, addCoords=True)
        # ensure the chiral centers are all defined
        Chem.AssignAtomChiralTagsFromStructure(rdkitmolh)

        ligand_mol = Molecule(rdkitmolh)
        return ligand_mol

    def create_system_generator(self, ligand_mol):
        forcefield_kwargs = {
            "constraints": app.HBonds,
            "rigidWater": True,
            "removeCMMotion": False,
            "hydrogenMass": 4 * unit.amu,
        }
        periodic_forcefield_kwargs = {"nonbondedMethod": app.PME}
        system_generator = SystemGenerator(
            forcefields=["amber/ff14SB.xml", "amber/tip3p_standard.xml"],
            small_molecule_forcefield=self.small_molecule_force_field,
            molecules=[ligand_mol],
            cache=None,
            forcefield_kwargs=forcefield_kwargs,
            periodic_forcefield_kwargs=periodic_forcefield_kwargs,
        )
        return system_generator, ligand_mol

    @staticmethod
    def get_complex_model(ligand_mol, protein_path):
        protein_pdb = PDBFile(str(protein_path))
        modeller = Modeller(protein_pdb.topology, protein_pdb.positions)
        # This next bit is black magic.
        # Modeller needs topology and positions. Lots of trial and error found that this is what works to get these from
        # an openforcefield Molecule object that was created from a RDKit molecule.
        # The topology part is described in the openforcefield API but the positions part grabs the first (and only)
        # conformer and passes it to Modeller. It works. Don't ask why!
        modeller.add(
            ligand_mol.to_topology().to_openmm(),
            ligand_mol.conformers[0].to_openmm(),
        )
        return modeller, ligand_mol

    @staticmethod
    def setup_and_solvate(system_generator, modeller, ligand_mol):
        # We need to temporarily create a Context in order to identify molecules for adding virtual bonds
        integrator = openmm.VerletIntegrator(1 * unit.femtoseconds)
        system = system_generator.create_system(modeller.topology, molecules=ligand_mol)
        context = openmm.Context(
            system, integrator, openmm.Platform.getPlatformByName("Reference")
        )
        molecules_atom_indices = context.getMolecules()
        del context, integrator, system

        # Solvate
        # we use the 'padding' option to define the periodic box. The PDB file does not contain any
        # unit cell information so we just create a box that has a 9A padding around the complex.
        modeller.addSolvent(
            system_generator.forcefield, model="tip3p", padding=12.0 * unit.angstroms
        )
        return modeller, molecules_atom_indices

    @staticmethod
    def create_system(system_generator, modeller, molecule_atom_indices, ligand_mol):
        mdtop = mdtraj.Topology.from_openmm(modeller.topology)
        output_indices = mdtop.select("not water")
        output_topology = mdtop.subset(output_indices).to_openmm()

        # Create the system using the SystemGenerator
        system = system_generator.create_system(modeller.topology, molecules=ligand_mol)

        # Add virtual bonds so solute is imaged together
        custom_bond_force = openmm.CustomBondForce("0")
        for molecule_index in range(len(molecule_atom_indices) - 1):
            custom_bond_force.addBond(
                molecule_atom_indices[molecule_index][0],
                molecule_atom_indices[molecule_index + 1][0],
                [],
            )
        system.addForce(custom_bond_force)

        return system, output_indices, output_topology

    def setup_simulation(
        self, modeller, system, output_indices, output_topology, outpath, platform
    ):
        # Add barostat

        system.addForce(MonteCarloBarostat(self._pressure, self._temperature))

        if self.rmsd_restraint:
            logger.info("Adding RMSD restraint")
            if self.rmsd_restraint_atom_indices:
                atom_indices = self.rmsd_restraint_atom_indices

            elif self.rmsd_restraint_type:
                if self.rmsd_restraint_type == "CA":
                    atom_indices = [
                        atom.index
                        for atom in modeller.topology.atoms()
                        if atom.residue.name not in solvent_types and atom.name == "CA"
                    ]

                elif self.rmsd_restraint_type == "heavy":
                    atom_indices = [
                        atom.index
                        for atom in modeller.topology.atoms()
                        if atom.residue.name not in solvent_types
                        and atom.element.name != "hydrogen"
                    ]
                    warnings.warn(
                        "Heavy atom RMSD restraint includes ligand atoms, are you sure this is what you want?"
                    )
            logger.debug(f"RMSD restraint atom indices: {atom_indices}")
            custom_cv_force = openmm.CustomCVForce("(K_RMSD/2)*(RMSD)^2")
            custom_cv_force.addGlobalParameter(
                "K_RMSD", self.rmsd_restraint_force_constant * 2
            )
            rmsd_force = openmm.RMSDForce(modeller.positions, atom_indices)
            custom_cv_force.addCollectiveVariable("RMSD", rmsd_force)
            system.addForce(custom_cv_force)
            logger.info("Added RMSD restraint force")

        integrator = LangevinMiddleIntegrator(
            self._temperature, self._collision_rate, self._timestep
        )

        simulation = Simulation(
            modeller.topology, system, integrator, platform=platform
        )
        context = simulation.context
        context.setPositions(modeller.positions)

        # Minimize energy
        simulation.minimizeEnergy()

        # Write minimized PDB
        output_positions = context.getState(
            getPositions=True, enforcePeriodicBox=False
        ).getPositions(asNumpy=True)
        with open(outpath / "minimized.pdb", "w") as outfile:
            PDBFile.writeFile(
                output_topology,
                output_positions[output_indices, :],
                file=outfile,
                keepIds=True,
            )
        return simulation, context

    def equilibrate(self, simulation):
        # Equilibrate
        simulation.context.setVelocitiesToTemperature(self._temperature)
        simulation.step(self.equilibration_steps)
        return simulation

    def run_production_simulation(
        self, simulation, context, output_indices, output_topology, outpath
    ):
        # Add reporter to generate XTC trajectory
        simulation.reporters.append(
            XTCReporter(
                str(outpath / "traj.xtc"),
                self.reporting_interval,
                atomSubset=output_indices,
            )
        )
        # Add reporter for rough timing info
        simulation.reporters.append(
            StateDataReporter(
                str(outpath / self.openmm_logname),
                self.reporting_interval,
                step=True,
                time=True,
                temperature=True,
                progress=True,
                remainingTime=True,
                speed=True,
                totalSteps=self.num_steps,
                separator="\t",
            )
        )
        logger.info(f"Running simulation for {self.num_steps} steps")
        if self.progressbar:
            pbar = tqdm(total=self.num_steps)
        for _ in range(self.n_snapshots):
            simulation.step(self.reporting_interval)
            if self.progressbar:
                pbar.update(self.reporting_interval)
        if self.progressbar:
            pbar.close()

        output_positions = context.getState(
            getPositions=True, enforcePeriodicBox=False
        ).getPositions(asNumpy=True)
        with open(outpath / "final.pdb", "w") as outfile:
            PDBFile.writeFile(
                output_topology,
                output_positions[output_indices, :],
                file=outfile,
                keepIds=True,
            )

        # Flush trajectories to force files to be closed
        for reporter in simulation.reporters:
            del reporter

        # Clean up to release GPU resources
        del simulation.context
        del simulation
        # return some sort of success/fail code
        return True

    def provenance(self) -> dict[str, str]:
        return {}
