import logging
from enum import Enum
from pathlib import Path
from typing import List  # noqa: F401
import dask
import mdtraj
import openmm
from asapdiscovery.data.logging import FileLogger
from mdtraj.reporters import XTCReporter
from openff.toolkit.topology import Molecule
from openmm import LangevinMiddleIntegrator, MonteCarloBarostat, app, unit
from openmm.app import Modeller, PDBFile, Simulation, StateDataReporter
from openmmforcefields.generators import SystemGenerator
from rdkit import Chem
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt

from asapdiscovery.simulation.simulate import OpenMMPlatform
from asapdiscovery.docking.docking_v2 import DockingResult
from asapdiscovery.docking.docking_data_validation import DockingResultCols
from asapdiscovery.data.dask_utils import actualise_dask_delayed_iterable
from asapdiscovery.data.schema_v2.ligand import Ligand
import pandas as pd

import abc


class SimulatorBase(BaseModel):
    """
    Base class for Simulators.
    """

    output_dir: Path = Field("md", description="Output directory")
    debug: bool = Field(False, description="Debug mode of the simulation")

    @abc.abstractmethod
    def _simulate(self) -> pd.DataFrame:
        ...

    def simulate(
        self,
        docking_results: list[DockingResult],
        use_dask: bool = False,
        dask_client=None,
        **kwargs,
    ) -> pd.DataFrame:
        if use_dask:
            delayed_outputs = []
            for res in docking_results:
                out = dask.delayed(self._simulate)(docking_results=[res], **kwargs)
                delayed_outputs.append(out)
            outputs = actualise_dask_delayed_iterable(
                delayed_outputs, dask_client, errors="raise"
            )
            outputs = [item for sublist in outputs for item in sublist]  # flatten
        else:
            outputs = self._visualize(docking_results=docking_results, **kwargs)

        return pd.DataFrame(outputs)

    @abc.abstractmethod
    def provenance(self) -> dict[str, str]:
        ...


class VanillaMDSimulatorV2(SimulatorBase):
    collision_rate: PositiveFloat = Field(
        1, description="Collision rate of the simulation"
    )
    openmm_logname: str = Field(
        "openmm_log.tsv", description="Name of the OpenMM log file"
    )
    openmm_platform: OpenMMPlatform = Field(
        OpenMMPlatform.Fastest, description="OpenMM platform to use"
    )
    temperature: PositiveFloat = Field(300, description="Temperature of the simulation")
    pressure: PositiveFloat = Field(1, description="Pressure of the simulation")
    timestep: PositiveFloat = Field(4, description="Timestep of the simulation")
    equilibration_steps: PositiveInt = Field(
        5000, description="Number of equilibration steps"
    )
    reporting_interval: PositiveInt = Field(
        1250, description="Reporting interval of the simulation"
    )
    num_steps: PositiveInt = Field(2500000, description="Number of simulation steps")

    class Config:
        arbitrary_types_allowed = True

    def _simulate(self, docking_results: list[DockingResult]) -> list[dict[str, str]]:
        self.temperature = self.temperature * unit.kelvin
        pressure = pressure * unit.atmospheres
        collision_rate = collision_rate / unit.picoseconds
        timestep = timestep * unit.femtoseconds
        equilibration_steps = equilibration_steps
        reporting_interval = reporting_interval
        num_steps = num_steps
        n_snapshots = int(self.num_steps / self.reporting_interval)
        num_steps = self.n_snapshots * self.reporting_interval

        # set platform
        platform = OpenMMPlatform(self.openmm_platform).get_platform()
        if platform.getName() == "CUDA" or platform.getName() == "OpenCL":
            platform.setPropertyDefaultValue("Precision", "mixed")

        if self.debug:
            self.platform = OpenMMPlatform.CPU.get_platform()

        data = []
        for result in docking_results:
            output_pref = result.get_combined_id()
            outpath = self.output_dir / output_pref
            if not outpath.exists():
                outpath.mkdir(parents=True)

            ligand_off_mol = self.process_ligand_rdkit(result.posed_ligand)
            system_generator, ligand_mol = self.create_system_generator(ligand_off_mol)
            modeller, ligand_mol = self.get_complex_model(ligand_mol, self.protein_path)
            modeller, mol_atom_indices = self.setup_and_solvate(
                system_generator, modeller, ligand_mol
            )
            system, output_indices, output_topology = self.create_system(
                system_generator, modeller, mol_atom_indices, ligand_off_mol
            )
            simulation, context = self.setup_simulation(
                modeller, system, output_indices, output_topology, outpath
            )
            simulation = self.equilibrate(simulation)
            retcode = self.run_production_simulation(
                simulation, context, output_indices, output_topology, outpath
            )
            row = {}
            row[
                DockingResultCols.LIGAND_ID.value
            ] = result.input_pair.ligand.compound_name
            row[
                DockingResultCols.TARGET_ID.value
            ] = result.input_pair.complex.target.target_name
            row[DockingResultCols.SMILES.value] = result.input_pair.ligand.smiles
            row[DockingResultCols.MD_PATH_TRAJ.value] = outpath / "traj.xtc"
            row[DockingResultCols.MD_PATH_MIN_PDB.value] = outpath / "minimized.pdb"
            row[DockingResultCols.MD_PATH_FINAL_PDB.value] = outpath / "final.pdb"
            row["md_success"] = retcode

            data.append(row)

        return data

    @staticmethod
    def process_ligand_rdkit(ligand: Ligand) -> Molecule:
        rdkitmol = Chem.MolFromSmiles(ligand.smiles)
        rdkitmolh = Chem.AddHs(rdkitmol, addCoords=True)
        # ensure the chiral centers are all defined
        Chem.AssignAtomChiralTagsFromStructure(rdkitmolh)

        ligand_mol = Molecule(rdkitmolh)
        return ligand_mol

    @staticmethod
    def create_system_generator(ligand_off_mol):
        forcefield_kwargs = {
            "constraints": app.HBonds,
            "rigidWater": True,
            "removeCMMotion": False,
            "hydrogenMass": 4 * unit.amu,
        }
        periodic_forcefield_kwargs = {"nonbondedMethod": app.PME}
        system_generator = SystemGenerator(
            forcefields=["amber/ff14SB.xml", "amber/tip3p_standard.xml"],
            small_molecule_forcefield="openff-1.3.1",
            molecules=[ligand_off_mol],
            cache=None,
            forcefield_kwargs=forcefield_kwargs,
            periodic_forcefield_kwargs=periodic_forcefield_kwargs,
        )
        return system_generator, ligand_off_mol

    @staticmethod
    def get_complex_model(ligand_off_mol, protein_path):
        protein_pdb = PDBFile(str(protein_path))
        modeller = Modeller(protein_pdb.topology, protein_pdb.positions)
        # This next bit is black magic.
        # Modeller needs topology and positions. Lots of trial and error found that this is what works to get these from
        # an openforcefield Molecule object that was created from a RDKit molecule.
        # The topology part is described in the openforcefield API but the positions part grabs the first (and only)
        # conformer and passes it to Modeller. It works. Don't ask why!
        modeller.add(
            ligand_off_mol.to_topology().to_openmm(),
            ligand_off_mol.conformers[0].to_openmm(),
        )
        return modeller, ligand_off_mol

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
        self, modeller, system, output_indices, output_topology, outpath
    ):
        # Add barostat

        system.addForce(MonteCarloBarostat(self.pressure, self.temperature))

        integrator = LangevinMiddleIntegrator(
            self.temperature, self.collision_rate, self.timestep
        )

        simulation = Simulation(
            modeller.topology, system, integrator, platform=self.platform
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
        simulation.context.setVelocitiesToTemperature(self.temperature)
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

        for _ in range(self.n_snapshots):
            simulation.step(self.reporting_interval)

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
