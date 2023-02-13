#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 18:10:15 2023
@author: kendalllemons
"""

import os
import openmm
import argparse
import rdkit

from openmm.app import PDBFile


## Parameters
def get_args():
    parser = argparse.ArgumentParser()

    ## Input arguments
    parser.add_argument(
        "-i", 
        "--input_pdb_path", 
        type = str,
        default = "/data/chodera/lemonsk/asap-datasets/openmm_setup_processed/prepped_receptor_0-processed.pdb",
        help = "Path to PDB file to simulate"
    )
    parser.add_argument(
        "-o", 
        "--output_dir",
        type = str,
        default = "/data/chodera/lemonsk/asap-datasets/prepped_mpro_P0009/",
        help="Output simulation directory."
    )
    parser.add_argument(
        "-l",
        "--ligand",
        type = str,
        default = "/data/chodera/lemonsk/covid-moonshot-ml/asapdiscovery/simulation/tests/inputs/MAT-POS-f2460aef-1.sdf",
        help = "Ligand"
    )
    parser.add_argument(
        "-p",
        "--protein",
        type = str,
        default = "/data/chodera/lemonsk/asap-datasets/openmm_setup_processed/prepped_receptor_0-processed.pdb",
        help = "Protein-receptor"
    )
    args = parser.parse_args()
    return args

args = get_args()

# Necessary Parameters

from openmm import unit, app
from openmm.app import PME, HBonds

nonbondedMethod = PME
nonbondedCutoff = 1.0*unit.nanometers
ewaldErrorTolerance = 0.000001
constraints = HBonds
rigidWater = True
constraintTolerance = 0.000001
hydrogenMass = 4.0*unit.amu

# Integration Options

dt = 4.0 * unit.femtoseconds
temperature = 300 * unit.kelvin
friction = 1.0 / unit.femtoseconds
pressure = 1.0 * unit.atmospheres
barostatInterval = 25

from rdkit import Chem
rdkitmol = Chem.SDMolSupplier(args.ligand)[0]
rdkitmolh = Chem.AddHs(rdkitmol, addCoords=True);
# ensure the chiral centers are all defined
Chem.AssignAtomChiralTagsFromStructure(rdkitmolh);
from openff.toolkit.topology import Molecule
ligand_mol = Molecule(rdkitmolh);

# Initialize a SystemGenerator
from openmmforcefields.generators import SystemGenerator
forcefield_kwargs = {'constraints': app.HBonds, 'rigidWater': True, 'removeCMMotion': False, 'hydrogenMass': 4*unit.amu }
periodic_forcefield_kwargs = {'nonbondedMethod': app.PME}
system_generator = SystemGenerator(
    forcefields=['amber/ff14SB.xml', 'amber/tip3p_standard.xml'],
    small_molecule_forcefield='openff-1.3.1',
    molecules=[ligand_mol], cache='cache.json',
    forcefield_kwargs=forcefield_kwargs, periodic_forcefield_kwargs=periodic_forcefield_kwargs);

# Use Modeller to combine the protein and ligand into a complex
from openmm.app import PDBFile
from openmm.app import Modeller
protein_pdb = PDBFile(args.protein)
modeller = Modeller(protein_pdb.topology, protein_pdb.positions)
# This next bit is black magic.
# Modeller needs topology and positions. Lots of trial and error found that this is what works to get these from
# an openforcefield Molecule object that was created from a RDKit molecule.
# The topology part is described in the openforcefield API but the positions part grabs the first (and only)
# conformer and passes it to Modeller. It works. Don't ask why!
modeller.add(ligand_mol.to_topology().to_openmm(), ligand_mol.conformers[0].to_openmm())


# We need to temporarily create a Context in order to identify molecules for adding virtual bonds
integrator = openmm.VerletIntegrator(1 * unit.femtoseconds)
system = system_generator.create_system(modeller.topology, molecules=ligand_mol)
context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName('CUDA'))
molecules_atom_indices = context.getMolecules()
del context, integrator, system

#Solvate
modeller.addSolvent(system_generator.forcefield, padding=0.9*unit.nanometers, model='tip3p')

print('Creation of System...')
#Creation of System
system = system_generator.create_system(modeller.topology, molecules=ligand_mol)

# Addition of Virtual Bonds
custom_bond_force = openmm.CustomBondForce('0')
for molecule_index in range(len(molecules_atom_indices)-1):
    custom_bond_force.addBond(molecules_atom_indices[molecule_index][0], molecules_atom_indices[molecule_index+1][0], [])
system.addForce(custom_bond_force)

print('Preparing the Simulation...')
# Prepare the Simulation
from openmm import LangevinMiddleIntegrator
from openmm.app import Simulation
from openmm import MonteCarloBarostat
from openmm import Platform

steps = 10000
equilibrationSteps = 100
platform = Platform.getPlatformByName('CUDA')

system.addForce(MonteCarloBarostat(pressure, temperature))
integrator = LangevinMiddleIntegrator(temperature, friction, dt)
simulation = Simulation(modeller.topology, system, integrator, platform=platform)
context = simulation.context
context.setPositions(modeller.positions)

print('System Configuration...')
# System Configuration

from openmm.app import DCDReporter
from openmm.app import StateDataReporter

# Minimize and Equilibrate
print('Performing energy minimization...')
simulation.minimizeEnergy()
print('Equilibrating...')
simulation.context.setVelocitiesToTemperature(temperature)
simulation.step(equilibrationSteps)

# Simulate
from mdtraj.reporters import XTCReporter
print('Simulating...')
dcdReporter = simulation.reporters.append(DCDReporter(os.path.join(args.output_dir,'trajectory.dcd'), 50))
dataReporter = simulation.reporters.append(StateDataReporter(os.path.join(args.output_dir,'log.txt'), 50, totalSteps=steps,
    step=True, speed=True, progress=True, potentialEnergy=True, temperature=True, separator='\t'))
xtcReporter = simulation.reporters.append(XTCReporter(os.path.join(args.output_dir,'trajectory.xtc'), 50))

simulation.currentStep = 0
simulation.step(steps)


print('Saving...')
positions = simulation.context.getState(getPositions=True).getPositions()
PDBFile.writeFile(simulation.topology, positions, open(os.path.join(args.output_dir,'output_P0009_final.pdb'), 'w'))
simulation.saveState(os.path.join(args.output_dir,'output.xml'))
print('Done')