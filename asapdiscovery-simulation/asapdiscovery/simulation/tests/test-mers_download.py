"""
@author: Alex Payne
"""
from asapdiscovery.simulation.utils import test_forcefield_generation
from pathlib import Path

paths = Path(
    "/lila/data/chodera/asap-datasets/mers_fauxalysis/20230307_prepped_mers_pdbs/"
).glob("*0_pdb")
if __name__ == "__main__":
    for path in paths:
        test_forcefield_generation(str(path))
