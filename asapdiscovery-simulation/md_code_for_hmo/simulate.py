"""
Runs a simulation with OpenMM.
"""
# Configure logging
import logging

# Set parameters for simulation
from openmm import unit
from rdkit import Chem

# from rich.logging import RichHandler

# FORMAT = "%(message)s"
# from rich.console import Console

# logging.basicConfig(
#     level=logging.INFO,
#     format=FORMAT,
#     datefmt="[%X]",
#     handlers=[RichHandler(markup=True)],
# )
# log = logging.getLogger("rich")


# define some standards.
temperature = 300 * unit.kelvin
pressure = 1 * unit.atmospheres
collision_rate = 1.0 / unit.picoseconds
timestep = 4.0 * unit.femtoseconds
equilibration_steps = 5000  # 20 ps
reporting_interval = 1250  # 5 ps

# some less standard parameters. These should be in CLI.
num_steps = 2500000  # 10ns; number of integrator steps
n_snapshots = (
    int(num_steps / reporting_interval) * reporting_interval
)  # recalculate number of steps to run

ligand_path = ""
protein_path = ""

# log.info(f":gear:  Processing {arguments['--receptor']} and {arguments['--ligand']}")
# log.info( f":clock1:  Will run {num_steps*timestep / unit.nanoseconds:3f} ns of production simulation to generate {n_snapshots} snapshots")


def set_platform():
    # could use structuring to increase flexibility
    # check whether we have a GPU platform and if so set the precision to mixed
    speed = 0
    from openmm import Platform

    for i in range(Platform.getNumPlatforms()):
        p = Platform.getPlatform(i)
        if p.getSpeed() > speed:
            platform = p
            speed = p.getSpeed()

    if platform.getName() == "CUDA" or platform.getName() == "OpenCL":
        platform.setPropertyDefaultValue("Precision", "mixed")
        # log.info(f":dart:  Setting precision for platform {platform.getName()} to mixed")


def create_system_generator():
    # this could do with some structuring to improve flexibility.

    # Initialize a SystemGenerator
    # log.info(":wrench:  Initializing SystemGenerator")
    from openmm import app
    from openmmforcefields.generators import SystemGenerator

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
        cache="cache.json",
        forcefield_kwargs=forcefield_kwargs,
        periodic_forcefield_kwargs=periodic_forcefield_kwargs,
    )
    return system_generator


def get_complex_model(ligand_path, protein_path):
    # load in ligand, protein, then combine them into an openmm object.

    # Read the molfile into RDKit, add Hs and create an openforcefield Molecule object
    # log.info(":pill:  Reading ligand")

    rdkitmol = Chem.SDMolSupplier(ligand_path)[0]
    # log.info(f":mage:  Adding hydrogens")
    rdkitmolh = Chem.AddHs(rdkitmol, addCoords=True)
    # ensure the chiral centers are all defined
    Chem.AssignAtomChiralTagsFromStructure(rdkitmolh)
    from openff.toolkit.topology import Molecule

    ligand_mol = Molecule(rdkitmolh)

    # Use Modeller to combine the protein and ligand into a complex
    # log.info(":cut_of_meat:  Reading protein")
    from openmm.app import PDBFile

    protein_pdb = PDBFile(protein_path)
    # log.info(":sandwich:  Preparing complex")
    from openmm.app import Modeller

    modeller = Modeller(protein_pdb.topology, protein_pdb.positions)
    # This next bit is black magic.
    # Modeller needs topology and positions. Lots of trial and error found that this is what works to get these from
    # an openforcefield Molecule object that was created from a RDKit molecule.
    # The topology part is described in the openforcefield API but the positions part grabs the first (and only)
    # conformer and passes it to Modeller. It works. Don't ask why!
    modeller.add(
        ligand_mol.to_topology().to_openmm(), ligand_mol.conformers[0].to_openmm()
    )

    return modeller


def setup_and_solvate(modeller):
    # We need to temporarily create a Context in order to identify molecules for adding virtual bonds
    # log.info(f":microscope:  Identifying molecules")
    import openmm

    integrator = openmm.VerletIntegrator(1 * unit.femtoseconds)
    system = system_generator.create_system(modeller.topology, molecules=ligand_mol)
    context = openmm.Context(
        system, integrator, openmm.Platform.getPlatformByName("Reference")
    )
    molecules_atom_indices = context.getMolecules()
    del context, integrator, system

    # Solvate
    # log.info(":droplet:  Adding solvent...")
    # we use the 'padding' option to define the periodic box. The PDB file does not contain any
    # unit cell information so we just create a box that has a 9A padding around the complex.
    modeller.addSolvent(
        system_generator.forcefield, model="tip3p", padding=9.0 * unit.angstroms
    )
    # log.info(":package:  System has %d atoms" % modeller.topology.getNumAtoms())

    return modeller


def create_system(modeller):
    # Determine which atom indices we want to use
    import mdtraj

    mdtop = mdtraj.Topology.from_openmm(modeller.topology)
    output_indices = mdtop.select("not water")
    output_topology = mdtop.subset(output_indices).to_openmm()

    # Create the system using the SystemGenerator
    # log.info(":globe_showing_americas:  Creating system...")
    system = system_generator.create_system(modeller.topology, molecules=ligand_mol)

    # Add virtual bonds so solute is imaged together
    # log.info(f":chains:  Adding virtual bonds between molecules")
    custom_bond_force = openmm.CustomBondForce("0")
    for molecule_index in range(len(molecules_atom_indices) - 1):
        custom_bond_force.addBond(
            molecules_atom_indices[molecule_index][0],
            molecules_atom_indices[molecule_index + 1][0],
            [],
        )
    system.addForce(custom_bond_force)

    return system, output_indices


def setup_simulation(modeller, system):
    # Add barostat
    from openmm import MonteCarloBarostat

    system.addForce(MonteCarloBarostat(pressure, temperature))
    # log.info(f":game_die: Default Periodic box:")
    # for dim in range(3):
    #     log.info(f"  :small_blue_diamond: {system.getDefaultPeriodicBoxVectors()[dim]}")

    # Create integrator
    # log.info(":building_construction:  Creating integrator...")
    from openmm import LangevinMiddleIntegrator

    integrator = LangevinMiddleIntegrator(temperature, collision_rate, timestep)

    # Create simulation
    # log.info(":mage:  Creating simulation...")
    from openmm.app import Simulation

    simulation = Simulation(modeller.topology, system, integrator, platform=platform)
    context = simulation.context
    context.setPositions(modeller.positions)

    # Minimize energy
    # log.info(":skier:  Minimizing ...")
    simulation.minimizeEnergy()

    # Write minimized PDB
    # log.info(f":page_facing_up:  Writing minimized PDB to {arguments['--minimized']}")
    output_positions = context.getState(
        getPositions=True, enforcePeriodicBox=False
    ).getPositions(asNumpy=True)
    with open("minimized.pdb", "w") as outfile:
        PDBFile.writeFile(
            output_topology,
            output_positions[output_indices, :],
            file=outfile,
            keepIds=False,
        )
    return simulation, context


def equilibrate(simulation):
    # Equilibrate
    # log.info(":fire:  Heating ...")
    simulation.context.setVelocitiesToTemperature(temperature)
    simulation.step(equilibration_steps)

    return simulation


def run_production_simulation(simulation, context, output_indices):
    # Add reporter to generate XTC trajectory
    # log.info(f":page_facing_up:  Will write XTC trajectory to {arguments['--xtctraj']}")
    from mdtraj.reporters import XTCReporter

    simulation.reporters.append(
        XTCReporter("traj.xtc", reporting_interval, atomSubset=output_indices)
    )

    # Run simulation
    # log.info(":coffee:  Starting simulation...")
    # from rich.progress import track

    # for snapshot_index in track(
    #     # range(n_snapshots), ":rocket: Running production simulation..."
    # ):
    #     simulation.step(reporting_interval)

    # Write final PDB
    # log.info(f":page_facing_up:  Writing final PDB to {arguments['--final']}")
    output_positions = context.getState(
        getPositions=True, enforcePeriodicBox=False
    ).getPositions(asNumpy=True)
    with open("final.pdb", "w") as outfile:
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


if __name__ == "__main__":
    set_platform()
    system_generator = create_system_generator()
    modeller = get_complex_model(ligand_path, protein_path)
    modeller = setup_and_solvate(modeller)
    system, output_indices = create_system(modeller)
    simulation, context = setup_simulation(modeller, system)
    simulation = equilibrate(simulation)
    run_production_simulation(simulation, context, output_indices)
