from asapdiscovery.data.schema import ExperimentalCompoundData


def test_classes():
    compound = ExperimentalCompoundData(compound_id="Test")
    assert compound.compound_id == "Test"
