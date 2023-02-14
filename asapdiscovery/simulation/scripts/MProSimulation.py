

import sys

# Configure logging
import logging
from rich.logging import RichHandler
FORMAT = "%(message)s"
from rich.console import Console
logging.basicConfig(
    level=logging.INFO, format=FORMAT, datefmt="[%X]", handlers=[RichHandler(markup=True)]
)
log = logging.getLogger("rich")

## Preliminary Imports
import os
import openmm
import argparse
import rdkit
from rdkit import Chem

## OpenMM Imports
from openmm.app import PDBFile
from openmm import unit, app
from openmm.app import PME, HBonds
from openmm.app import Modeller
from openmm import LangevinMiddleIntegrator
from openmm.app import Simulation
from openmm import MonteCarloBarostat
from openmm import Platform
from openmm.app import DCDReporter
from openmm.app import Simulation
import mdtraj
from mdtraj.reporters import XTCReporter
from openff.toolkit.topology import Molecule
from openmmforcefields.generators import SystemGenerator

#Simulation Parameters

steps = 10000
equilibrationSteps = 100
reportingInterval = 1250
dt = 4.0 * unit.femtoseconds
temperature = 300 * unit.kelvin
friction = 1.0 / unit.femtoseconds
pressure = 1.0 * unit.atmospheres
barostatInterval = 25
nonbondedMethod = PME
nonbondedCutoff = 1.0*unit.nanometers
ewaldErrorTolerance = 0.000001
constraints = HBonds
rigidWater = True
constraintTolerance = 0.000001
hydrogenMass = 4.0*unit.amu
platform = Platform.getPlatformByName('CUDA')

__doc__ = """Generate explicit-solvent molecular dynamics simulation for MPro Files.
Usage:
  MProSimulation.py --protein=STR --ligand=STR --nsteps=INT [--selection=SELECT] [--output_dir=STR]
  MProSimulation.py (-h | --help)
Options:
  -h --help           Show this screen.
  --protein=STR       Receptor PDB file path
  --ligand=STR        Ligand SDF file path.
  --nsteps=INT        Number of steps to run.
  --selection=SELECT  MDTraj selection to use (e.g. 'not water') [default: all].
  --ouput_dir=STR     Output directory for reporters
"""

from docopt import docopt
args = docopt(__doc__, version='MPro Simulation 0.1')

num_steps = int(args['--nsteps']) # number of integrator steps
n_snapshots = int(num_steps / reportingInterval) # calculate number of snapshots that will be generated
num_steps = n_snapshots * reportingInterval # recalculate number of steps to run

# Read the molfile into RDKit, add Hs and create an openforcefield Molecule object
log.info(':pill:  Reading ligand')
rdkitmol = Chem.SDMolSupplier(args['--ligand'])[0]
log.info(f':mage:  Adding hydrogens')
rdkitmolh = Chem.AddHs(rdkitmol, addCoords=True);
# ensure the chiral centers are all defined
Chem.AssignAtomChiralTagsFromStructure(rdkitmolh);
ligand_mol = Molecule(rdkitmolh);

# Initialize a SystemGenerator
forcefield_kwargs = {'constraints': app.HBonds, 'rigidWater': True, 'removeCMMotion': False, 'hydrogenMass': 4*unit.amu }
periodic_forcefield_kwargs = {'nonbondedMethod': app.PME}
system_generator = SystemGenerator(
    forcefields=['amber/ff14SB.xml', 'amber/tip3p_standard.xml'],
    small_molecule_forcefield='openff-1.3.1',
    molecules=[ligand_mol], cache='cache.json',
    forcefield_kwargs=forcefield_kwargs, periodic_forcefield_kwargs=periodic_forcefield_kwargs);

# Use Modeller to combine the protein and ligand into a complex
log.info(':cut_of_meat:  Reading protein')
protein_pdb = PDBFile(args['--protein'])
log.info(':sandwich:  Preparing complex')
modeller = Modeller(protein_pdb.topology, protein_pdb.positions)
# This next bit is black magic.
# Modeller needs topology and positions. Lots of trial and error found that this is what works to get these from
# an openforcefield Molecule object that was created from a RDKit molecule.
# The topology part is described in the openforcefield API but the positions part grabs the first (and only)
# conformer and passes it to Modeller. It works. Don't ask why!
modeller.add(ligand_mol.to_topology().to_openmm(), ligand_mol.conformers[0].to_openmm())

# We need to temporarily create a Context in order to identify molecules for adding virtual bonds
log.info(f':microscope:  Identifying molecules')
integrator = openmm.VerletIntegrator(1*unit.femtoseconds)
system = system_generator.create_system(modeller.topology, molecules=ligand_mol)
context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName('CUDA'))
molecules_atom_indices = context.getMolecules()
del context, integrator, system

# Solvate
log.info(':droplet:  Adding solvent...')
# we use the 'padding' option to define the periodic box. The PDB file does not contain any
# unit cell information so we just create a box that has a 9A padding around the complex.
modeller.addSolvent(system_generator.forcefield, model='tip3p', padding=0.9*unit.nanometers)
log.info(':package:  System has %d atoms' % modeller.topology.getNumAtoms())

# Determine which atom indices we want to use
mdtop = mdtraj.Topology.from_openmm(modeller.topology)
atom_selection = args['--selection']
log.info(f':clipboard:  Using selection: {atom_selection}')
output_indices = mdtop.select(atom_selection)
output_topology = mdtop.subset(output_indices).to_openmm()

# Add virtual bonds so solute is imaged together
log.info(f':chains:  Adding virtual bonds between molecules')
custom_bond_force = openmm.CustomBondForce('0')
for molecule_index in range(len(molecules_atom_indices)-1):
    custom_bond_force.addBond(molecules_atom_indices[molecule_index][0], molecules_atom_indices[molecule_index+1][0], [])
system.addForce(custom_bond_force)

# Add barostat
system.addForce(MonteCarloBarostat(pressure, temperature));
log.info(f':game_die: Default Periodic box:')
for dim in range(3):
    log.info(f'  :small_blue_diamond: {system.getDefaultPeriodicBoxVectors()[dim]}')

# Create integrator
log.info(':building_construction:  Creating integrator...')
integrator = LangevinMiddleIntegrator(temperature, friction, dt)

# Create simulation
log.info(':mage:  Creating simulation...')
simulation = Simulation(modeller.topology, system, integrator, platform=platform)
context = simulation.context
context.setPositions(modeller.positions)

# Minimize energy
log.info(':skier:  Minimizing ...')
simulation.minimizeEnergy();

# Equilibrate
log.info(':fire:  Heating ...')
simulation.context.setVelocitiesToTemperature(temperature);
simulation.step(equilibrationSteps);

# Reporters
log.info(':gold mining: Runninng Reporters...')
dcdReporter = simulation.reporters.append(DCDReporter(os.path.join(args['--output_dir'], args['--protein'].split('_')[0]+'.pdb'), reportingInterval))
xtcReporter = simulation.reporters.append(XTCReporter(os.path.join(args['--output_dir'], args['--protein'].split('_')[0]+'.xtc'), reportingInterval))

# Run simulation
log.info(':coffee:  Starting simulation...')
from rich.progress import track
for snapshot_index in track(range(n_snapshots), ':rocket: Running production simulation...'):
    simulation.step(reportingInterval)