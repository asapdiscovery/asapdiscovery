# Configure logging
import logging
from pathlib import Path
from typing import List  # noqa: F401

import mdtraj
import openmm
import tqdm
from asapdiscovery.data.logging import FileLogger
from mdtraj.reporters import XTCReporter
from openff.toolkit.topology import Molecule
from openmm import LangevinMiddleIntegrator, MonteCarloBarostat, Platform, app, unit
from openmm.app import Modeller, PDBFile, Simulation, StateDataReporter
from openmmforcefields.generators import SystemGenerator
from rdkit import Chem


class VanillaMDSimulator:
    def __init__(
        self,
        ligand_paths: list[Path],
        protein_path: Path,
        temperature: float = 300,
        pressure: float = 1,
        collision_rate: float = 1,
        timestep: float = 4,
        equilibration_steps: int = 5000,
        reporting_interval: int = 1250,
        num_steps: int = 2500000,
        output_paths: list[Path] = None,
        logger: FileLogger = None,
        openmm_logname: str = "openmm_log.tsv",
        debug: bool = False,
    ):
        self.ligand_paths = ligand_paths
        self.protein_path = protein_path
        # thermo
        self.temperature = temperature * unit.kelvin
        self.pressure = pressure * unit.atmospheres
        self.collision_rate = collision_rate / unit.picoseconds
        self.timestep = timestep * unit.femtoseconds
        self.equilibration_steps = equilibration_steps
        self.reporting_interval = reporting_interval
        self.num_steps = num_steps
        self.n_snapshots = int(self.num_steps / self.reporting_interval)
        self.num_steps = self.n_snapshots * self.reporting_interval
        self.openmm_logname = openmm_logname

        if output_paths is None:
            outdir = Path("md").mkdir(exist_ok=True)
            self.output_paths = [outdir / ligand.parent for ligand in ligand_paths]
        else:
            self.output_paths = output_paths

        # init
        if logger is None:
            self.logger = FileLogger(
                "md_log.txt", "./", stdout=True, level=logging.INFO
            ).getLogger()
        else:
            self.logger = logger

        self.logger.info("Starting MD run")
        self.debug = debug
        if self.debug:
            self.logger.info("Running in debug mode")
            self.logger.SetLevel(logging.DEBUG)
        self.logger.debug(f"Running MD on {len(self.ligand_paths)} ligands")
        self.logger.debug(f"Running MD on {self.protein_path} protein")
        self.logger.debug(f"Writing to  {self.output_paths}")

        self.set_platform()

    def set_platform(self):
        # could use structuring to increase flexibility
        # check whether we have a GPU platform and if so set the precision to mixed
        self.logger.info("Setting platform for MD run")
        speed = 0

        if Platform.getNumPlatforms() == 0:
            raise ValueError("No compatible OpenMM patforms detected")

        for i in range(Platform.getNumPlatforms()):
            p = Platform.getPlatform(i)
            if p.getSpeed() > speed:
                platform = p
                speed = p.getSpeed()

        if platform.getName() == "CUDA" or platform.getName() == "OpenCL":
            platform.setPropertyDefaultValue("Precision", "mixed")
            self.logger.info(
                f"Setting precision for platform {platform.getName()} to mixed"
            )

        self.logger.info("Using platform {platform.getName()}")
        self.platform = platform
        if self.debug:
            self.logger.debug("Setting platform to CPU for debugging")
            self.platform = Platform.getPlatformByName("CPU")

    def process_ligand(self, ligand_path) -> Molecule:
        self.logger.debug("Prepping ligand")
        rdkitmol = Chem.SDMolSupplier(str(ligand_path))[0]
        rdkitmolh = Chem.AddHs(rdkitmol, addCoords=True)
        # ensure the chiral centers are all defined
        Chem.AssignAtomChiralTagsFromStructure(rdkitmolh)

        ligand_mol = Molecule(rdkitmolh)
        return ligand_mol

    def create_system_generator(self, ligand_mol, outpath):
        self.logger.debug("Initializing SystemGenerator")
        self.logger.debug(f"Creating system generator for {ligand_mol}")
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
            molecules=[ligand_mol],
            cache=None,
            forcefield_kwargs=forcefield_kwargs,
            periodic_forcefield_kwargs=periodic_forcefield_kwargs,
        )
        return system_generator, ligand_mol

    def get_complex_model(self, ligand_mol, protein_path):
        # load in ligand, protein, then combine them into an openmm object.
        self.logger.debug(f"Creating complex model for {ligand_mol} and {protein_path}")
        # Use Modeller to combine the protein and ligand into a complex
        self.logger.debug("Reading protein")

        protein_pdb = PDBFile(str(protein_path))
        self.logger.debug("Preparing complex")

        modeller = Modeller(protein_pdb.topology, protein_pdb.positions)
        # This next bit is black magic.
        # Modeller needs topology and positions. Lots of trial and error found that this is what works to get these from
        # an openforcefield Molecule object that was created from a RDKit molecule.
        # The topology part is described in the openforcefield API but the positions part grabs the first (and only)
        # conformer and passes it to Modeller. It works. Don't ask why!
        modeller.add(
            ligand_mol.to_topology().to_openmm(), ligand_mol.conformers[0].to_openmm()
        )
        return modeller, ligand_mol

    def setup_and_solvate(self, system_generator, modeller, ligand_mol):
        # We need to temporarily create a Context in order to identify molecules for adding virtual bonds
        self.logger.debug("Setup and solvate")
        integrator = openmm.VerletIntegrator(1 * unit.femtoseconds)
        system = system_generator.create_system(modeller.topology, molecules=ligand_mol)
        context = openmm.Context(
            system, integrator, openmm.Platform.getPlatformByName("Reference")
        )
        molecules_atom_indices = context.getMolecules()
        del context, integrator, system

        # Solvate
        self.logger.debug("Adding solvent...")
        # we use the 'padding' option to define the periodic box. The PDB file does not contain any
        # unit cell information so we just create a box that has a 9A padding around the complex.
        modeller.addSolvent(
            system_generator.forcefield, model="tip3p", padding=9.0 * unit.angstroms
        )
        self.logger.info(f"System has {modeller.topology.getNumAtoms()} atoms")
        return modeller, molecules_atom_indices

    def create_system(
        self, system_generator, modeller, molecule_atom_indices, ligand_mol
    ):
        self.logger.debug("Creating system...")
        # Determine which atom indices we want to use

        mdtop = mdtraj.Topology.from_openmm(modeller.topology)
        output_indices = mdtop.select("not water")
        output_topology = mdtop.subset(output_indices).to_openmm()

        # Create the system using the SystemGenerator
        system = system_generator.create_system(modeller.topology, molecules=ligand_mol)

        # Add virtual bonds so solute is imaged together
        self.logger.debug("Adding virtual bonds between molecules")
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
        self.logger.debug("Default Periodic box:")
        for dim in range(3):
            self.logger.info(f" {system.getDefaultPeriodicBoxVectors()[dim]}")

        # Create integrator
        self.logger.info("Creating integrator...")

        integrator = LangevinMiddleIntegrator(
            self.temperature, self.collision_rate, self.timestep
        )

        # Create simulation
        self.logger.info("Creating simulation...")

        simulation = Simulation(
            modeller.topology, system, integrator, platform=self.platform
        )
        context = simulation.context
        context.setPositions(modeller.positions)

        # Minimize energy
        self.logger.info("Minimizing ...")
        simulation.minimizeEnergy()

        # Write minimized PDB
        self.logger.debug("Writing minimized PDB")
        output_positions = context.getState(
            getPositions=True, enforcePeriodicBox=False
        ).getPositions(asNumpy=True)
        with open(outpath / "minimized.pdb", "w") as outfile:
            PDBFile.writeFile(
                output_topology,
                output_positions[output_indices, :],
                file=outfile,
                keepIds=False,
            )
        return simulation, context

    def equilibrate(self, simulation):
        # Equilibrate
        self.logger.info("Starting equilibration...")
        simulation.context.setVelocitiesToTemperature(self.temperature)
        simulation.step(self.equilibration_steps)
        self.logger.info("Finished")

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

        # Run simulation
        self.logger.info("Running simulation...")

        for snapshot_index in range(self.n_snapshots):
            simulation.step(self.reporting_interval)

        self.logger.info("Finished")

        # Write final PDB
        self.logger.info("Writing final PDB")
        output_positions = context.getState(
            getPositions=True, enforcePeriodicBox=False
        ).getPositions(asNumpy=True)
        with open(outpath / "final.pdb", "w") as outfile:
            PDBFile.writeFile(
                output_topology,
                output_positions[output_indices, :],
                file=outfile,
                keepIds=False,
            )

        # Flush trajectories to force files to be closed
        for reporter in simulation.reporters:
            del reporter

        # Clean up to release GPU resources
        del simulation.context
        del simulation
        # return some sort of success/fail code
        return 0

    def run_simulation(self, ligand, outpath):
        if not outpath.exists():
            outpath.mkdir(parents=True)
        self.logger.info(
            f"starting simulation for {ligand} writing simulation to {outpath}"
        )
        processed_ligand = self.process_ligand(ligand)
        system_generator, ligand_mol = self.create_system_generator(
            processed_ligand, outpath
        )
        modeller, ligand_mol = self.get_complex_model(ligand_mol, self.protein_path)
        modeller, mol_atom_indices = self.setup_and_solvate(
            system_generator, modeller, ligand_mol
        )
        system, output_indices, output_topology = self.create_system(
            system_generator, modeller, mol_atom_indices, processed_ligand
        )
        simulation, context = self.setup_simulation(
            modeller, system, output_indices, output_topology, outpath
        )
        simulation = self.equilibrate(simulation)
        retcode = self.run_production_simulation(
            simulation, context, output_indices, output_topology, outpath
        )
        return retcode

    def run_all_simulations(self):
        retcodes = []
        for ligand, outpath in zip(self.ligand_paths, self.output_paths):
            retcode = self.run_simulation(ligand, outpath)
            retcodes.append(retcode)
        return retcodes
