import pytest
from asapdiscovery.data.testing.test_resources import fetch_test_file
from asapdiscovery.data.expand_stereo import StereoExpanderOptions, StereoExpander
from asapdiscovery.data.openeye import load_openeye_smi


@pytest.fixture(scope="session")
def chalcogran_defined():
    # test file with two stereocenters defined
    smi_file = fetch_test_file("chalcogran_defined.smi")
    return smi_file


def test_expand_default_construct():
    options = StereoExpanderOptions()
    expander = StereoExpander(options)
    assert expander.options.warts == False
    assert expander.options.force_flip == False
    assert expander.options.postera_names == True
    assert expander.options.debug == False


def test_expand_from_mol(chalcogran_defined):
    expander = StereoExpander(StereoExpanderOptions())
    mol = load_openeye_smi(chalcogran_defined)
    expanded_mols = expander.expand_mol(mol)
    assert len(expanded_mols) == 1


def test_expand_from_file(chalcogran_defined):
    expander = StereoExpander(StereoExpanderOptions())
    expanded_mols = expander.expand_structure_file(chalcogran_defined)
    assert len(expanded_mols) == 1


def test_expand_from_file_force_flip(chalcogran_defined):
    expander = StereoExpander(StereoExpanderOptions(force_flip=True, warts=True))
    expanded_mols = expander.expand_structure_file(chalcogran_defined)
    assert len(expanded_mols) == 4
    for i, mol in enumerate(expanded_mols):
        # check the titles have warts
        assert mol.GetTitle() == f"_{i + 1}"


@pytest.mark.parametrize("outfile", ["test.mol2", "test.sdf", "test.smi"])
def test_expand_from_file_with_output(chalcogran_defined, outfile, tmp_path):
    expander = StereoExpander(StereoExpanderOptions())
    expanded_mols = expander.expand_structure_file(
        chalcogran_defined, outfile=tmp_path / outfile
    )
    assert len(expanded_mols) == 1
