from asapdiscovery.data.pdb import download_pdb_structure, load_pdbs_from_yaml
from pathlib import Path


pdb_yaml = Path("../../../metadata/mers-structures.yaml")
pdb_dict = load_pdbs_from_yaml(str(pdb_yaml))
pdb_dir = Path("pdb_download")
pdb_dir.mkdir(exist_ok=True)

download_pdb_structure("8DGY", pdb_dir, pdb_type="pdb")
download_pdb_structure("8DGY", pdb_dir, pdb_type="cif")
download_pdb_structure("8DGY", pdb_dir, pdb_type="cif1")
