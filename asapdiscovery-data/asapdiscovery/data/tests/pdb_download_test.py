from pytest import skip

skip("CI for file download not implemented yet", allow_module_level=True)
from pathlib import Path

from asapdiscovery.data.pdb import download_pdb_structure, load_pdbs_from_yaml

pdb_yaml = Path("../../../../metadata/mers-structures.yaml")
pdb_dict = load_pdbs_from_yaml(str(pdb_yaml))
pdb_dir = Path("pdb_download")
pdb_dir.mkdir(exist_ok=True)

download_pdb_structure("8DGY", pdb_dir, file_format="pdb")
download_pdb_structure("8DGY", pdb_dir, file_format="cif")
download_pdb_structure("8DGY", pdb_dir, file_format="cif1")
