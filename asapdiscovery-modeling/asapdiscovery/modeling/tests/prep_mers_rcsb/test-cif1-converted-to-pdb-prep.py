from asapdiscovery.docking.modeling import (
    remove_extra_ligands,
    mutate_residues,
    align_receptor,
    prep_receptor,
)
from asapdiscovery.data.schema import CrystalCompoundData
from asapdiscovery.data.utils import seqres_to_res_list
from asapdiscovery.data.openeye import (
    load_openeye_cif,
    save_openeye_pdb,
    load_openeye_pdb,
)
from pathlib import Path
import yaml

out_path = Path("")
in_path = Path("inputs")
# pdb_path = in_path / "rcsb_8DGY-assembly1.pdb"
pdb_path = in_path / "rcsb_8DGY-assembly1-openmm.pdb"
ref_path = in_path / "reference.pdb"
seqres_path = Path("../../../../../metadata/mpro_mers_seqres.yaml")
with open(seqres_path) as f:
    seqres_dict = yaml.safe_load(f)
# seqres = seqres_dict["SEQRES"]
pdb_id = "8DGY"
active_site_chain = "A"

mol = load_openeye_pdb(pdb_path.as_posix())

## Align protein
## TODO: This currently only keeps waters for one of the chains
## it would be nice to have both
mol = align_receptor(
    initial_complex=mol,
    ref_prot=ref_path.as_posix(),
    dimer=True,
    split_initial_complex=False,
    mobile_chain=active_site_chain,
    ref_chain="A",
)
save_openeye_pdb(mol, str(out_path / "align_test.pdb"))

## Delete extra copies of ligand in the complex
# removed = remove_extra_ligands(mol, lig_chain=active_site_chain)

## TODO: this doesn't work for these structures because the ligand isn't named "LIG"
## Fine for now since we want to get rid of the ligand anyway
## but it would be nice to do this intelligently


# res_list = seqres_to_res_list(seqres)
# print("Mutating to provided seqres")
# print(res_list)

# Mutate the residues to match the residue list
# mol = mutate_residues(mol, res_list)
# save_openeye_pdb(mol, str(out_path / "mutate_test.pdb"))

# Prep receptor
print("Prepping receptor")
design_units = prep_receptor(
    mol,
    site_residue=f"HIS:166: :{active_site_chain}",
    # loop_db=loop_db,
    protein_only=True,
    # seqres=" ".join(res_list),
)

# xtal = CrystalCompoundData(
#     str_fn=pdb_path.as_posix(),
#     output_name=f"{pdb_id}_0{active_site_chain}",
#     active_site_chain=active_site_chain,
#     # active_site=f"HIS:41: :{active_site_chain}",
#     # chains=["A", "B"],
#     # oligomeric_state="dimer",
#     # protein_chains=["A", "B"],
# )
#
# prep_mp(
#     xtal=xtal,
#     ref_prot=ref_path.as_posix(),
#     seqres=seqres_path.as_posix(),
#     out_base=out_path.as_posix(),
#     loop_db=None,
#     protein_only=False,
# )
