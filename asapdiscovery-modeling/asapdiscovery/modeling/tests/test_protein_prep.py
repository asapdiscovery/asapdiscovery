# This test suite can be run with a local path to save the output, ie:
# pytest test_protein_prep.py --local_path=/path/to/save/files
# without a local path, output files will be saved to a temporary directory
# This behaviour is controlled by the output_dir fixture.
from pathlib import Path

import pytest
from asapdiscovery.data.openeye import oechem
from asapdiscovery.data.testing.test_resources import fetch_test_file
from asapdiscovery.modeling.modeling import protein_prep_workflow
from asapdiscovery.modeling.schema import PrepOpts


@pytest.fixture
def ref():
    return fetch_test_file("reference.pdb")


@pytest.fixture
def reference_output_files():
    return {
        "sars": {
            "protein": fetch_test_file("Mpro-P2660_0A_bound-prepped_protein.pdb"),
            "ligand": fetch_test_file("Mpro-P2660_0A_bound-prepped_ligand.sdf"),
            "complex": fetch_test_file("Mpro-P2660_0A_bound-prepped_complex.pdb"),
            "design_unit": fetch_test_file("Mpro-P2660_0A_bound-prepped_receptor.oedu"),
        },
        "mers": {
            "protein": fetch_test_file("rcsb_8czv-assembly1-prepped_protein.pdb"),
            "design_unit": fetch_test_file("rcsb_8czv-assembly1-prepped_receptor.oedu"),
        },
    }


def test_output_file_download(reference_output_files):
    for target, files in reference_output_files.items():
        for component, fn in files.items():
            assert Path(fn).exists()
            assert Path(fn).is_file()


@pytest.fixture
def loop_db():
    return fetch_test_file("fragalysis-mpro_spruce.loop_db")


@pytest.fixture
def prep_dict(mers_target, sars_target):
    return {
        "mers": (mers_target, fetch_test_file("mpro_mers_seqres.yaml")),
        "sars": (sars_target, fetch_test_file("mpro_sars2_seqres.yaml")),
    }


class TestProteinPrep:
    @pytest.mark.parametrize("target_name", ["mers", "sars"])
    def test_protein_prep_workflow(
        self,
        target_name,
        prep_dict,
        ref,
        output_dir,
        loop_db,
        reference_output_files,
        ref_chain="A",
    ):
        target, seqres_yaml = prep_dict[target_name]

        prep_opts = PrepOpts(
            ref_fn=ref,
            ref_chain=ref_chain,
            loop_db=loop_db,
            seqres_yaml=seqres_yaml,
            output_dir=output_dir,
        )
        prepped_target = protein_prep_workflow(target, prep_opts)

        generated_output_files = {}
        if "protein" in prepped_target.molecule_filter.components_to_keep:
            generated_output_files["protein"] = prepped_target.protein
            generated_output_files["design_unit"] = prepped_target.design_unit
        if "ligand" in prepped_target.molecule_filter.components_to_keep:
            generated_output_files["ligand"] = prepped_target.ligand
            generated_output_files["complex"] = prepped_target.complex
        for component, fn in generated_output_files.items():
            assert Path(fn).exists()
            assert Path(fn).is_file()

        # Load the prepared design unit
        du = oechem.OEDesignUnit()
        oechem.OEReadDesignUnit(str(prepped_target.design_unit), du)

        assert type(du) == oechem.OEDesignUnit
        assert du.HasReceptor()
        if "ligand" in prepped_target.molecule_filter.components_to_keep:
            assert du.HasLigand()
        # TODO: Find a better way to test if the output matches the excepted output in
        #  reference_output_files, as hashing does not work due to
        #  any minor changes in atom positions causing a different hash
        #  Things to check could include:
        #  - topologies are the same
        #  - the correct components are in the correct chain
        #    using the find_component_chains function as used in test_modeling_utils
        #  - receptor can be added to POSIT
