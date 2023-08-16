from pathlib import Path

import pytest
from asapdiscovery.data.openeye import load_openeye_cif1, load_openeye_pdb, oechem
from asapdiscovery.data.schema import CrystalCompoundData
from asapdiscovery.data.testing.test_resources import fetch_test_file
from asapdiscovery.modeling.schema import MoleculeFilter, PreppedTarget, PreppedTargets


def pytest_addoption(parser):
    parser.addoption(
        "--local_path",
        type=str,
        default=None,
        help="If provided, use this path to output files for tests",
    )


@pytest.fixture(scope="session")
def local_path(request):
    return request.config.getoption("--local_path")


# This needs to have a scope of session so that a new tmp file is not created for each test
@pytest.fixture(scope="session")
def output_dir(tmp_path_factory, local_path):
    if not type(local_path) is str:
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
        oe_active_site_residue="HIS:41: :A",
        molecule_filter=MoleculeFilter(components_to_keep=["protein"]),
    )


@pytest.fixture
def target_dataset(sars_target, mers_target):
    target_dataset = PreppedTargets.from_list([sars_target, mers_target])
    return target_dataset


@pytest.fixture
def sars_oe(sars):
    # Load structure
    prot = load_openeye_pdb(str(sars))
    assert isinstance(prot, oechem.OEGraphMol)
    return prot


@pytest.fixture
def mers_oe(mers):
    # Load structure
    prot = load_openeye_cif1(str(mers))
    assert isinstance(prot, oechem.OEGraphMol)
    return prot


@pytest.fixture
def oemol_dict(sars_oe, mers_oe):
    return {"sars": sars_oe, "mers": mers_oe}
