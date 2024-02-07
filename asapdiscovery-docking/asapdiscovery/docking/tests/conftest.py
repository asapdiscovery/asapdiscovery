from pathlib import Path

import pytest
from asapdiscovery.data.openeye import oechem
from asapdiscovery.data.schema_v2.complex import PreppedComplex
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.testing.test_resources import fetch_test_file
from asapdiscovery.docking.docking import DockingInputMultiStructure, DockingInputPair
from asapdiscovery.docking.openeye import POSITDocker


@pytest.fixture()
def local_path(request):
    try:
        return request.config.getoption("--local_path")
    except ValueError:
        return None


# This needs to have a scope of session so that a new tmp file is not created for each test
@pytest.fixture()
def output_dir(tmp_path_factory, local_path):
    if type(local_path) is not str:
        return tmp_path_factory.mktemp("test_prep")
    else:
        local_path = Path(local_path)
        local_path.mkdir(exist_ok=True)
        assert local_path.exists()
        return local_path


@pytest.fixture()
def ligand():
    return Ligand.from_sdf(
        fetch_test_file("Mpro-P0008_0A_ERI-UCB-ce40166b-17.sdf"), compound_name="test"
    )


@pytest.fixture()
def ligand_simple():
    return Ligand.from_smiles("CCCOCO", compound_name="test2")


@pytest.fixture()
def prepped_complex():
    return PreppedComplex.from_oedu_file(
        fetch_test_file("Mpro-P2660_0A_bound-prepped_receptor.oedu"),
        ligand_kwargs={"compound_name": "test"},
        target_kwargs={"target_name": "test", "target_hash": "mock_hash"},
    )


@pytest.fixture()
def prepped_complexes():
    cached_dus = {
        "Mpro-x1002": "du_cache/Mpro-x1002_0A_bound.oedu",
        "Mpro-x0354": "du_cache/Mpro-x0354_0A_bound.oedu",
    }
    return [
        PreppedComplex.from_oedu_file(
            fetch_test_file(cached_du),
            ligand_kwargs={"compound_name": "test"},
            target_kwargs={"target_name": name, "target_hash": "mock_hash"},
        )
        for name, cached_du in cached_dus.items()
    ]


@pytest.fixture()
def docking_input_pair(ligand, prepped_complex):
    return DockingInputPair(complex=prepped_complex, ligand=ligand)


@pytest.fixture()
def docking_input_pair_simple(ligand_simple, prepped_complex):
    return DockingInputPair(complex=prepped_complex, ligand=ligand_simple)


@pytest.fixture()
def docking_multi_structure(prepped_complexes, ligand):
    return DockingInputMultiStructure(complexes=prepped_complexes, ligand=ligand)


@pytest.fixture()
def results(docking_input_pair):
    docker = POSITDocker()
    results = docker.dock([docking_input_pair])
    return results


@pytest.fixture()
def results_simple(docking_input_pair_simple):
    docker = POSITDocker()
    results = docker.dock([docking_input_pair_simple])
    return results


@pytest.fixture()
def results_multi(results, results_simple):
    return results + results_simple


@pytest.fixture()
def mol_with_constrained_confs() -> oechem.OEMol:
    """Load a multiconformer OEMol from an sdf"""
    mol = oechem.OEMol()
    ifs = oechem.oemolistream(
        str(fetch_test_file("constrained_conformer/ASAP-0008650.sdf"))
    )
    ifs.SetConfTest(oechem.OEIsomericConfTest())
    oechem.OEReadMolecule(ifs, mol)
    return mol


@pytest.fixture()
def mac1_complex():
    return PreppedComplex.parse_file(
        fetch_test_file("constrained_conformer/complex.json")
    )
