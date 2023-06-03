# This test suite can be run with a local path to save the output, ie:
# pytest test_protein_prep.py --local_path=/path/to/save/files
# without a local path, output files will be saved to a temporary directory
# This behaviour is controlled by the prepped_files fixture.
from pathlib import Path

import pytest
from asapdiscovery.data.openeye import oechem
from asapdiscovery.data.schema import CrystalCompoundData
from asapdiscovery.data.testing.test_resources import fetch_test_file
from asapdiscovery.modeling.modeling import protein_prep_workflow
from asapdiscovery.modeling.schema import (
    MoleculeFilter,
    PrepOpts,
    PreppedTarget,
    PreppedTargets,
)

# This needs to have a scope of session so that a new tmp file is not created for each test
@pytest.fixture(scope="session")
def prepped_files(tmp_path_factory, local_path):
    if not type(local_path) == str:
        return tmp_path_factory.mktemp("test_prep")
    else:
        local_path = Path(local_path)
        local_path.mkdir(exist_ok=True)
        assert local_path.exists()
        return local_path


@pytest.fixture
def sars():
    return fetch_test_file("Mpro-P2660_0A_bound.pdb")


@pytest.fixture
def mers():
    return fetch_test_file("rcsb_8czv-assembly1.cif")


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
def sars_xtal(sars):
    return CrystalCompoundData(
        str_fn=str(sars),
    )


@pytest.fixture
def sars_target(sars_xtal):
    return PreppedTarget(
        source=sars_xtal,
        active_site_chain="A",
        output_name=Path(sars_xtal.str_fn).stem,
        molecule_filter=MoleculeFilter(
            components_to_keep=["protein", "ligand"], ligand_chain="A"
        ),
    )


@pytest.fixture
def mers_xtal(mers):
    return CrystalCompoundData(
        str_fn=str(mers),
    )


@pytest.fixture
def mers_target(mers_xtal):
    return PreppedTarget(
        source=mers_xtal,
        active_site_chain="A",
        output_name=Path(mers_xtal.str_fn).stem,
        active_site="HIS:41: :A:0: ",
        molecule_filter=MoleculeFilter(components_to_keep=["protein"]),
    )


@pytest.fixture
def target_dataset(sars_target, mers_target):
    target_dataset = PreppedTargets.from_list([sars_target, mers_target])
    print(target_dataset)
    return target_dataset


class TestCrystalCompoundDataset:
    def test_dataset_creation(self, target_dataset, sars_target, mers_target):
        assert len(target_dataset.iterable) == 2
        assert target_dataset.iterable[0].source.str_fn == str(
            sars_target.source.str_fn
        )
        assert target_dataset.iterable[1].source.str_fn == str(
            mers_target.source.str_fn
        )

    @pytest.mark.skip(
        reason="Multiple embedded schema objects need more logic to serialize to csv"
    )
    def test_dataset_csv_usage(
        self, target_dataset, prepped_files, csv_name="to_prep.csv"
    ):
        to_prep_csv = prepped_files / csv_name
        target_dataset.to_csv(to_prep_csv)
        assert to_prep_csv.exists()
        assert to_prep_csv.is_file()

        dataset = PreppedTargets.from_csv(prepped_files / csv_name)
        assert dataset == target_dataset

        dataset.iterable[0].active_site_chain = "B"
        assert dataset != target_dataset

    def test_dataset_pickle(
        self, target_dataset, prepped_files, pkl_name="to_prep.pkl"
    ):
        pkl_file = prepped_files / pkl_name
        target_dataset.to_pkl(pkl_file)
        assert pkl_file.exists()
        assert pkl_file.is_file()

        loaded_dataset = PreppedTargets.from_pkl(pkl_file)

        assert loaded_dataset == target_dataset

    @pytest.mark.skip(
        reason="Multiple embedded schema objects need more logic to serialize to json"
    )
    def test_dataset_json(
        self, target_dataset, prepped_files, json_name="to_prep.json"
    ):
        json_file = prepped_files / json_name
        target_dataset.to_json(json_file)
        assert json_file.exists()
        assert json_file.is_file()

        loaded_dataset = PreppedTargets.from_json(json_file)

        assert loaded_dataset == target_dataset


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
        prepped_files,
        loop_db,
        reference_output_files,
        ref_chain="A",
    ):
        target, seqres_yaml = prep_dict[target_name]
        print(target, seqres_yaml)
        prep_opts = PrepOpts(
            ref_fn=ref,
            ref_chain=ref_chain,
            loop_db=loop_db,
            seqres_yaml=seqres_yaml,
            output_dir=prepped_files,
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
