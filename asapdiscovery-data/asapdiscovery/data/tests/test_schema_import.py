from asapdiscovery.data.schema import (
    ExperimentalCompoundData,
    CrystalCompoundData,
    CrystalCompoundDataset,
)


def test_classes(tmp_path):
    compound = ExperimentalCompoundData(compound_id="Test")
    assert compound.compound_id == "Test"

    xtal = CrystalCompoundData(compound_id="Test")
    assert xtal.compound_id == "Test"

    csv_path = tmp_path / "test.csv"
    dataset = CrystalCompoundDataset.from_list([xtal])
    dataset.to_csv(csv_path)

    loaded_dataset = CrystalCompoundDataset.from_csv(csv_path)
    assert csv_path.exists()
    assert loaded_dataset.iterable[0].compound_id == "Test"
    assert loaded_dataset == dataset
