"""
Objects and functions to test the structures handling capabilities of asapdiscovery-data package

That is downloading, processing and storing PDB/CIF or similar structure files.
"""

import pathlib

import pkg_resources
import pytest
from asapdiscovery.data.services.rcsb.rcsb_download import download_pdb_structure


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

    def test_load_save_openeye_pdb(self, tmp_path):
        """Test that a downloaded pdb file can be loaded with the openeye-specific functions"""
        from asapdiscovery.data.backend.openeye import (
            load_openeye_pdb,
            save_openeye_pdb,
        )

        pdb_id = "8DGY"
        file_path = download_pdb_structure(pdb_id, tmp_path, file_format="pdb")
        mol = load_openeye_pdb(file_path)
        out_path = f"{tmp_path}/test_oe_save.pdb"
        save_openeye_pdb(mol, out_path)
        assert pathlib.Path(out_path).is_file(), "Could not save OE PDB file."


# TODO: Finish test once we have a good example data
@pytest.mark.skip(reason="No example input required for test yet.")
class TestLigands:
    """Class to test ligand specific functionality, such as ligand filtering."""

    def test_ligand_filtering(self):
        """Test SMARTS pattern matching ligand filtering"""
        from asapdiscovery.data.util import utils

        # First, parse the fragalysis directory and
        csv_file = "CSV_FILE_NEEDED_HERE.csv"
        fragalysis_dir = "FRAGALYSIS_DIR_NEEDED_HERE"
        sars_xtals = utils.parse_fragalysis_data(csv_file, fragalysis_dir)

        # For the compounds for which we have smiles strings, get a dictionary mapping the Compound_ID to the smiles
        cmp_to_smiles_dict = {
            compound_id: data.smiles
            for compound_id, data in sars_xtals.items()
            if data.smiles
        }

        smarts_queries_csv = pkg_resources.resource_filename(
            "asapdiscovery.data", "data/smarts_queries.csv"
        )

        # Filter based on the smiles using this OpenEye function
        filtered_inputs = utils.filter_docking_inputs(
            smarts_queries=smarts_queries_csv,
            docking_inputs=cmp_to_smiles_dict,
        )

        # Get a new dictionary of sars xtals based on the filtered inputs
        print(filtered_inputs)
        sars_xtals_filtered = {
            compound_id: data
            for compound_id, data in sars_xtals.items()
            if compound_id in filtered_inputs
        }
        print(sars_xtals_filtered)
        print(len(sars_xtals_filtered))
