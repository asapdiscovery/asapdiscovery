import pytest
from asapdiscovery.data.openeye import oechem
from asapdiscovery.data.schema_v2.complex import PreppedComplex
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.data.testing.test_resources import fetch_test_file
from asapdiscovery.docking.docking import DockingInputMultiStructure, DockingInputPair
from asapdiscovery.docking.openeye import POSITDockingResults


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
def results():  # precomputed results
    return [POSITDockingResults.from_json_file(fetch_test_file("docking_results.json"))]


@pytest.fixture()
def results_simple():
    return [
        POSITDockingResults.from_json_file(
            fetch_test_file("docking_results_simple.json")
        )
    ]


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
