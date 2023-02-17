from pathlib import Path
from asapdiscovery.data.openeye import (
    load_openeye_pdb,
    save_openeye_pdb,
    openeye_perceive_residues,
    split_openeye_design_unit,
    split_openeye_mol,
)

inputs = Path("inputs")
cifpath = inputs / "rcsb_8DGY-assembly1.cif"
output = Path("outputs")

print("Loading cif and writing to pdb file")
from openmm.app import PDBxFile, PDBFile

cif = PDBxFile(str(cifpath))

outfile = output / f"{cifpath.stem}-00.pdb"

## the keep ids flag is critical to make sure the residue numbers are correct
with open(outfile, "w") as f:
    PDBFile.writeFile(cif.topology, cif.positions, f, keepIds=True)

print("Loading pdb to OpenEye")
prot = load_openeye_pdb(str(outfile))

print("Aligning to ref")
from asapdiscovery.docking.modeling import align_receptor

ref_path = inputs / f"reference.pdb"
prot = align_receptor(
    initial_complex=prot,
    ref_prot=ref_path.as_posix(),
    dimer=True,
    split_initial_complex=True,
    ref_chain="A",
)
aligned = str(output / f"{cifpath.stem}-01.pdb")
save_openeye_pdb(prot, aligned)

print("Preparing Sprucing options")
loop_path = Path(
    "/Users/alexpayne/Scientific_Projects/mers-drug-discovery/spruce_bace.loop_db"
)
import yaml

seqres_path = Path("../../../../../metadata/mpro_mers_seqres.yaml")
with open(seqres_path) as f:
    seqres_dict = yaml.safe_load(f)
seqres = seqres_dict["SEQRES"]

from asapdiscovery.data.utils import seqres_to_res_list

res_list = seqres_to_res_list(seqres)
seqres = " ".join(res_list)

print("Making mutations")
from asapdiscovery.docking.modeling import mutate_residues

prot = mutate_residues(prot, res_list, place_h=True)

print("Sprucing protein")
from asapdiscovery.modeling.modeling import spruce_protein

du = spruce_protein(
    initial_prot=prot, seqres=seqres, loop_db=str(loop_path), return_du=True
)
print("Saving Design Unit")
from openeye import oechem

du_fn = output / f"{cifpath.stem}-02.oedu"
oechem.OEWriteDesignUnit(str(du_fn), du)

print("Saving PDB")
from openeye import oechem

prot = oechem.OEGraphMol()
du.GetProtein(prot)
prot_fn = output / f"{cifpath.stem}-02.pdb"
save_openeye_pdb(prot, str(prot_fn))
