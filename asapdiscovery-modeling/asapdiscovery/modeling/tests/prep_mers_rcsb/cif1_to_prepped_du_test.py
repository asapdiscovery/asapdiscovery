from pathlib import Path

import yaml
from asapdiscovery.data.openeye import load_openeye_pdb, oechem, save_openeye_pdb
from asapdiscovery.data.utils import seqres_to_res_list
from asapdiscovery.modeling.modeling import (
    align_receptor,
    mutate_residues,
    spruce_protein,
)
from openmm.app import PDBFile, PDBxFile

inputs = Path("inputs")
cifpath = inputs / "rcsb_8DGY-assembly1.cif"
output = Path("outputs")

print("Loading cif and writing to pdb file")

cif = PDBxFile(str(cifpath))

outfile = output / f"{cifpath.stem}-00.pdb"

# the keep ids flag is critical to make sure the residue numbers are correct
with open(outfile, "w") as f:
    PDBFile.writeFile(cif.topology, cif.positions, f, keepIds=True)

print("Loading pdb to OpenEye")
prot = load_openeye_pdb(str(outfile))

print("Aligning to ref")

ref_path = inputs / "reference.pdb"
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


seqres_path = Path("../../../../../metadata/mpro_mers_seqres.yaml")
with open(seqres_path) as f:
    seqres_dict = yaml.safe_load(f)
seqres = seqres_dict["SEQRES"]


res_list = seqres_to_res_list(seqres)
sequence = " ".join(res_list)

print("Making mutations")


prot = mutate_residues(prot, res_list, place_h=True)

print("Sprucing protein")


du = spruce_protein(
    initial_prot=prot, seqres=sequence, loop_db=str(loop_path), return_du=True
)
print("Saving Design Unit")


du_fn = output / f"{cifpath.stem}-02.oedu"
oechem.OEWriteDesignUnit(str(du_fn), du)

print("Saving PDB")


prot = oechem.OEGraphMol()
du.GetProtein(prot)

# Add SEQRES entries if they're not present
if (not oechem.OEHasPDBData(prot, "SEQRES")) and seqres:
    for seqres_line in seqres.split("\n"):
        if seqres_line != "":
            oechem.OEAddPDBData(prot, "SEQRES", seqres_line[6:])

prot_fn = output / f"{cifpath.stem}-02.pdb"
save_openeye_pdb(prot, str(prot_fn))
