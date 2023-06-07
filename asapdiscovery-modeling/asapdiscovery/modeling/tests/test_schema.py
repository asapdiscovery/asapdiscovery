from pathlib import Path

import pydantic
import pytest
from asapdiscovery.data.openeye import oechem
from asapdiscovery.data.schema import CrystalCompoundData
from asapdiscovery.data.testing.test_resources import fetch_test_file
from asapdiscovery.modeling.modeling import save_design_unit
from asapdiscovery.modeling.schema import MoleculeFilter, PreppedTarget, PreppedTargets


class TestMoleculeFilter:
    @pytest.mark.parametrize(
        "components",
        [["ligand"], ["protein", "ligand"], ["protein", "ligand", "water"]],
    )
    def test_molecule_filter_component_success(self, components):
        MoleculeFilter(components_to_keep=components)

    @pytest.mark.parametrize(
        "components",
        [["prot"], ["basket", "ligand"], "help"],
    )
    def test_molecule_filter_component_failure(self, components):
        with pytest.raises(ValueError):
            MoleculeFilter(components_to_keep=components)

    @pytest.mark.parametrize(
        "ligand_chain",
        [list("A")],
    )
    def test_molecule_filter_ligand_chain_failure(self, ligand_chain):
        with pytest.raises((ValueError, pydantic.ValidationError)):
            MoleculeFilter(ligand_chain=ligand_chain)


def test_prepped_target(output_dir):
    du_path = fetch_test_file("Mpro-P2660_0A_bound-prepped_receptor.oedu")
    du = oechem.OEDesignUnit()
    oechem.OEReadDesignUnit(str(du_path), du)
    prepped_target = PreppedTarget(
        source=CrystalCompoundData(),
        output_name="test",
        molecule_filter=MoleculeFilter(components_to_keep=["protein", "ligand"]),
    )
    assert prepped_target.prepped is False

    prepped_target.set_prepped()
    assert prepped_target.prepped

    prepped_target.output_dir = output_dir / prepped_target.output_name

    saved_target = save_design_unit(du, prepped_target)
    for fn in [saved_target.ligand, saved_target.complex, saved_target.protein]:
        assert Path(fn).exists()
        assert Path(fn).is_file()


class TestPreppedTargets:
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
        self, target_dataset, output_dir, csv_name="to_prep.csv"
    ):
        to_prep_csv = output_dir / csv_name
        target_dataset.to_csv(to_prep_csv)
        assert to_prep_csv.exists()
        assert to_prep_csv.is_file()

        dataset = PreppedTargets.from_csv(output_dir / csv_name)
        assert dataset == target_dataset

        dataset.iterable[0].active_site_chain = "B"
        assert dataset != target_dataset

    def test_dataset_pickle(self, target_dataset, output_dir, pkl_name="to_prep.pkl"):
        pkl_file = output_dir / pkl_name
        target_dataset.to_pkl(pkl_file)
        assert pkl_file.exists()
        assert pkl_file.is_file()

        loaded_dataset = PreppedTargets.from_pkl(pkl_file)

        assert loaded_dataset == target_dataset

    @pytest.mark.skip(
        reason="Multiple embedded schema objects need more logic to serialize to json"
    )
    def test_dataset_json(self, target_dataset, output_dir, json_name="to_prep.json"):
        json_file = output_dir / json_name
        target_dataset.to_json(json_file)
        assert json_file.exists()
        assert json_file.is_file()

        loaded_dataset = PreppedTargets.from_json(json_file)

        assert loaded_dataset == target_dataset
