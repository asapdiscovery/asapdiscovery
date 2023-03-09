# Configure logging
import logging
import multiprocessing as mp
import sys
from pathlib import Path

from rich.logging import RichHandler

FORMAT = "%(message)s"
from rich.console import Console

logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler(markup=True)],
)
log = logging.getLogger("rich")

import argparse

## Preliminary Imports
import os
import re

import mdtraj
import openmm
import rdkit
from mdtraj.reporters import XTCReporter
from openff.toolkit.topology import Molecule
from openmm import LangevinMiddleIntegrator, MonteCarloBarostat, Platform, app, unit

## OpenMM Imports
from openmm.app import (
    PME,
    DCDReporter,
    HBonds,
    Modeller,
    PDBFile,
    Simulation,
    StateDataReporter,
)
from openmmforcefields.generators import SystemGenerator
from rdkit import Chem

# Simulation Parameters

steps = 10000
equilibrationSteps = 100
reportingInterval = 1250
dt = 4.0 * unit.femtoseconds
temperature = 300 * unit.kelvin
friction = 1.0 / unit.femtoseconds
pressure = 1.0 * unit.atmospheres
barostatInterval = 25
nonbondedMethod = PME
nonbondedCutoff = 1.0 * unit.nanometers
ewaldErrorTolerance = 0.000001
constraints = HBonds
rigidWater = True
constraintTolerance = 0.000001
hydrogenMass = 4.0 * unit.amu
platform = Platform.getPlatformByName("CUDA")


def get_args():
    parser = argparse.ArgumentParser()

    ## Input arguments
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        default="/data/chodera/asap-datasets/current/sars_01_prepped_v3/",
        help="Path to PDB and SDF files",
    )
    parser.add_argument(
        "-p",
        "--protein",
        type=str,
        default="/data/chodera/lemonsk/asap-datasets/openmm_setup_processed/prepped_receptor_0-processed.pdb",
        help="Path to PDB file to simulate",
    )
    parser.add_argument(
        "-l",
        "--ligand",
        type=str,
        default="/data/chodera/lemonsk/covid-moonshot-ml/asapdiscovery/simulation/tests/inputs/MAT-POS-f2460aef-1.sdf",
        help="Ligand",
    )
    parser.add_argument("-n", "--nsteps", type=int, default=12, help="Number of Steps")
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="/data/chodera/lemonsk/asap-datasets/MPro_Simulations/",
        help="Output simulation directory.",
    )
    args = parser.parse_args()
    return args


def simulate_mpro(input_dir, protein, ligand, nsteps, output_dir):
    ## Necessary for Reporters
    save = re.search(r"Mpro-P+\d{4}", protein).group(0)
    logfile = output_dir / f"{save}-log.txt"
    logger = logging.getLogger(save)
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(str(logfile), mode="w")
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    num_steps = int(nsteps)  # number of integrator steps
    n_snapshots = int(
        num_steps / reportingInterval
    )  # calculate number of snapshots that will be generated
    num_steps = n_snapshots * reportingInterval  # recalculate number of steps to run

    # Read the molfile into RDKit, add Hs and create an openforcefield Molecule object
    logger.info(":pill:  Reading ligand")
    rdkitmol = Chem.SDMolSupplier(ligand)[0]
    logger.info(f":mage:  Adding hydrogens")
    rdkitmolh = Chem.AddHs(rdkitmol, addCoords=True)
    # ensure the chiral centers are all defined
    Chem.AssignAtomChiralTagsFromStructure(rdkitmolh)
    ligand_mol = Molecule(rdkitmolh)

    # Initialize a SystemGenerator
    logger.info(":factory: System Combination")
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

    # Use Modeller to combine the protein and ligand into a complex
    logger.info(":cut_of_meat:  Reading protein")
    protein_pdb = PDBFile(protein)
    logger.info(":sandwich:  Preparing complex")
    modeller = Modeller(protein_pdb.topology, protein_pdb.positions)
    # This next bit is black magic.
    # Modeller needs topology and positions. Lots of trial and error found that this is what works to get these from
    # an openforcefield Molecule object that was created from a RDKit molecule.
    # The topology part is described in the openforcefield API but the positions part grabs the first (and only)
    # conformer and passes it to Modeller. It works. Don't ask why!
    modeller.add(
        ligand_mol.to_topology().to_openmm(), ligand_mol.conformers[0].to_openmm()
    )

    # We need to temporarily create a Context in order to identify molecules for adding virtual bonds
    logger.info(f":microscope:  Identifying molecules")
    integrator = openmm.VerletIntegrator(1 * unit.femtoseconds)
    system = system_generator.create_system(modeller.topology, molecules=ligand_mol)
    context = openmm.Context(
        system, integrator, openmm.Platform.getPlatformByName("CUDA")
    )
    molecules_atom_indices = context.getMolecules()
    del context, integrator, system

    # Solvate
    logger.info(":droplet:  Adding solvent...")
    # we use the 'padding' option to define the periodic box. The PDB file does not contain any
    # unit cell information so we just create a box that has a 9A padding around the complex.
    modeller.addSolvent(
        system_generator.forcefield, model="tip3p", padding=0.9 * unit.nanometers
    )
    logger.info(":package:  System has %d atoms" % modeller.topology.getNumAtoms())

    # Create the system using the SystemGenerator
    logger.info(":globe_showing_americas:  Creating system...")
    system = system_generator.create_system(modeller.topology, molecules=ligand_mol)

    # Add virtual bonds so solute is imaged together
    logger.info(f":chains:  Adding virtual bonds between molecules")
    custom_bond_force = openmm.CustomBondForce("0")
    for molecule_index in range(len(molecules_atom_indices) - 1):
        custom_bond_force.addBond(
            molecules_atom_indices[molecule_index][0],
            molecules_atom_indices[molecule_index + 1][0],
            [],
        )
    system.addForce(custom_bond_force)

    # Add barostat
    system.addForce(MonteCarloBarostat(pressure, temperature))
    logger.info(f":game_die: Default Periodic box:")
    for dim in range(3):
        logger.info(
            f"  :small_blue_diamond: {system.getDefaultPeriodicBoxVectors()[dim]}"
        )

    # Create integrator
    logger.info(":building_construction:  Creating integrator...")
    integrator = LangevinMiddleIntegrator(temperature, friction, dt)

    # Create simulation
    logger.info(":mage:  Creating simulation...")
    simulation = Simulation(modeller.topology, system, integrator, platform=platform)
    context = simulation.context
    context.setPositions(modeller.positions)

    # Minimize energy
    logger.info(":skier:  Minimizing ...")
    simulation.minimizeEnergy()

    # Equilibrate
    logger.info(":fire:  Heating ...")
    simulation.context.setVelocitiesToTemperature(temperature)
    simulation.step(equilibrationSteps)

    # logger.info(':fencer: Creating New Output Directory')
    # new_directory = os.mkdir(os.path.join(output_dir + save))

    # Reporters
    logger.info(":steam_locomotive: Running Simulation...")
    dataReporter = simulation.reporters.append(
        StateDataReporter(
            os.path.join(output_dir + save + "_simulation_" + "log.txt"),
            50,
            totalSteps=steps,
            step=True,
            speed=True,
            progress=True,
            potentialEnergy=True,
            temperature=True,
            separator="\t",
        )
    )
    xtcReporter = simulation.reporters.append(
        XTCReporter(
            os.path.join(output_dir + save + "_simulation_" + ".xtc"), reportingInterval
        )
    )

    simulation.currentStep = 0
    simulation.step(steps)

    logger.info(":CD: Saving...")
    positions = simulation.context.getState(getPositions=True).getPositions()
    PDBFile.writeFile(
        simulation.topology,
        positions,
        open(os.path.join(output_dir + save + "_simulation" + "_final.pdb"), "w"),
    )
    simulation.saveState(os.path.join(output_dir, "output.xml"))
    print("Done")

    def main():
        args = get_args()

        protein = Path(args.protein)
        proteinpaths = protein.glob("*protein.pdb")
        ligand = Path(args.ligand)
        ligandpaths = ligand.glob("*.sdf")
        output = Path(args.output_dir)
        output.mkdir(exist_ok=True)

        mpro_args = [
            [proteinpaths, ligandpaths, output]
            for proteinpaths, ligandpaths in input_dir
        ]
        nprocs = min(mp.cpu_count(), len(mpro_args), args.num_cores)
        print(f"Prepping {len(mpro_args)} structures over {nprocs} cores.")
        with mp.Pool(processes=nprocs) as pool:
            pool.starmap(simulate_mpro, mpro_args)

    if __name__ == "__main__":
        main()
