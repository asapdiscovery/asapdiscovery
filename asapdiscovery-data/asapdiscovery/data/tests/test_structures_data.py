"""
Objects and functions to test the structures handling capabilities of asapdiscovery-data package

That is downloading, processing and storing PDB/CIF or similar structure files.
"""

import pathlib

import pkg_resources
import pytest
from asapdiscovery.data.pdb import download_pdb_structure


@pytest.fixture
def pdbs_from_yaml():
    """Fixture that gets pdb dictionary from mers-structures.yaml file"""
    from asapdiscovery.data.pdb import load_pdbs_from_yaml

    yaml_file = pkg_resources.resource_filename(
        "asapdiscovery-data", "data/mers-structures.yaml"
    )
    pdb_dict = load_pdbs_from_yaml(yaml_file)
    return pdb_dict


@pytest.mark.skip(reason="No OE License in CI yet.")
class TestAsapPDB:
    """Class for testing PDB munging in ASAP discovery data package"""

    def test_download_pdb(self, tmp_path):
        """Test default behavior to download PDB file"""
        pdb_id = "8DGY"
        file_path = pathlib.Path(download_pdb_structure(pdb_id, tmp_path))
        assert file_path.is_file(), f"Could not download {pdb_id} pdb file."

    def test_download_cif(self, tmp_path):
        """Test downloading CIF file"""
        pdb_id = "8DGY"
        file_path = pathlib.Path(
            download_pdb_structure(pdb_id, tmp_path, file_format="cif")
        )
        assert file_path.is_file(), f"Could not download {pdb_id} cif file."

    def test_download_cif_assembly(self, tmp_path):
        """Test downloading CIF assembly file"""
        pdb_id = "8DGY"
        file_path = pathlib.Path(
            download_pdb_structure(pdb_id, tmp_path, file_format="cif1")
        )
        assert file_path.is_file(), f"Could not download {pdb_id} cif assembly file."

    @pytest.mark.parametrize("file_format", ["pdb", "cif", "cif1"])
    def test_load_save_openeye_pdb(self, tmp_path, file_format):
        """Test that a downloaded pdb file can be loaded with the openeye-specific functions"""
        from asapdiscovery.data.openeye import load_openeye_pdb, save_openeye_pdb

        pdb_id = "8DGY"
        file_path = download_pdb_structure(pdb_id, tmp_path, file_format=file_format)
        mol = load_openeye_pdb(file_path)
        out_path = f"{tmp_path}/test_oe_save.pdb"
        save_openeye_pdb(mol, out_path)
        assert pathlib.Path(out_path).is_file(), "Could not save OE PDB file."
