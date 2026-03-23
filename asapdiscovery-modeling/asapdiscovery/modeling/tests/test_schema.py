import os

import pydantic
import pytest
from pydantic import ValidationError

from asapdiscovery.data.backend.openeye import load_openeye_design_unit
from asapdiscovery.data.schema.complex import Complex
from asapdiscovery.data.schema.schema_base import MoleculeFilter
from asapdiscovery.data.schema.target import TargetIdentifiers
from asapdiscovery.modeling.schema import PreppedComplex, PreppedTarget


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


# PreppedTarget tests


def test_preppedtarget_from_oedu_file(oedu_file):
    pt = PreppedTarget.from_oedu_file(
        oedu_file, target_name="PreppedTargetTestName", target_hash="mock-hash"
    )
    oedu = pt.to_oedu()
    assert oedu.GetTitle() == "(AB) > LIG(A-403)"  # from one of the old files


def test_preppedtarget_from_oedu_file_at_least_one_id(oedu_file):
    with pytest.raises(ValidationError):
        # neither id is set
        PreppedTarget.from_oedu_file(oedu_file)


def test_preppedtarget_to_pdb_file(oedu_file, tmpdir):
    """Make sure a target can be saved to pdb file for vis"""

    with tmpdir.as_cwd():
        pt = PreppedTarget.from_oedu_file(
            oedu_file, target_name="PreppedTargetTest", target_hash="mock-hash"
        )
        file_name = "test_protein.pdb"
        pt.to_pdb_file(file_name)
        assert os.path.exists(file_name) is True


def test_preppedtarget_from_oedu_file_at_least_one_target_id(oedu_file):
    with pytest.raises(ValidationError):
        _ = PreppedTarget.from_oedu_file(oedu_file, ids=TargetIdentifiers())


def test_prepped_target_from_oedu_file_bad_file():
    with pytest.raises(FileNotFoundError):
        # neither id is set
        _ = PreppedTarget.from_oedu_file(
            "bad_file", target_name="PreppedTargetTestName"
        )


def test_prepped_target_from_oedu(oedu_file):
    loaded_oedu = load_openeye_design_unit(oedu_file)
    loaded_oedu.SetTitle("PreppedTargetTestName")
    pt = PreppedTarget.from_oedu(
        loaded_oedu, target_name="PreppedTargetTestName", target_hash="mock-hash"
    )
    oedu = pt.to_oedu()
    assert oedu.GetTitle() == "PreppedTargetTestName"


def test_prepped_target_from_oedu_file_roundtrip(oedu_file, tmp_path):
    pt = PreppedTarget.from_oedu_file(
        oedu_file, target_name="PreppedTargetTestName", target_hash="mock-hash"
    )
    pt.to_oedu_file(tmp_path / "test.oedu")
    pt2 = PreppedTarget.from_oedu_file(
        tmp_path / "test.oedu",
        target_name="PreppedTargetTestName",
        target_hash="mock-hash",
    )
    # these two comparisons should be the same
    assert pt == pt2
    assert pt.data_equal(pt2)


def test_prepped_target_from_oedu_roundtrip(oedu_file):
    pt = PreppedTarget.from_oedu_file(
        oedu_file, target_name="PreppedTargetTestName", target_hash="mock-hash"
    )
    du = pt.to_oedu()
    pt2 = PreppedTarget.from_oedu(
        du, target_name="PreppedTargetTestName", target_hash="mock-hash"
    )
    # these two comparisons should be the same
    assert pt == pt2
    assert pt.data_equal(pt2)


def test_prepped_target_json_roundtrip(oedu_file):
    pt = PreppedTarget.from_oedu_file(
        oedu_file, target_name="PreppedTargetTestName", target_hash="mock-hash"
    )
    js = pt.model_dump_json()
    pt2 = PreppedTarget.from_json(js)
    # these two comparisons should be the same
    assert pt == pt2
    assert pt.data_equal(pt2)
    du = pt2.to_oedu()
    assert du.GetTitle() == "(AB) > LIG(A-403)"


def test_prepped_target_json_file_roundtrip(oedu_file, tmp_path):
    pt = PreppedTarget.from_oedu_file(
        oedu_file, target_name="PreppedTargetTestName", target_hash="mock-hash"
    )
    path = tmp_path / "test.json"
    pt.to_json_file(path)
    pt2 = PreppedTarget.from_json_file(path)
    # these two comparisons should be the same
    assert pt == pt2
    assert pt.data_equal(pt2)
    du = pt2.to_oedu()
    assert du.GetTitle() == "(AB) > LIG(A-403)"


# PreppedComplex tests


def test_prepped_complex_from_complex(complex_pdb):
    c1 = Complex.from_pdb(
        complex_pdb,
        target_kwargs={"target_name": "test"},
        ligand_kwargs={"compound_name": "test"},
    )
    c2 = PreppedComplex.from_complex(c1, prep_kwargs={})
    du = c2.target.to_oedu()
    assert du.HasReceptor()
    assert du.HasLigand()
    assert c2.target.target_name == "test"
    assert c2.ligand.compound_name == "test"


def test_prepped_complex_from_oedu_file(complex_oedu):
    c = PreppedComplex.from_oedu_file(
        complex_oedu,
        target_kwargs={"target_name": "test", "target_hash": "test hash"},
        ligand_kwargs={"compound_name": "test"},
    )
    assert c.target.target_name == "test"
    assert c.ligand.compound_name == "test"


def test_prepped_complex_hash(complex_pdb):
    comp = Complex.from_pdb(
        complex_pdb,
        target_kwargs={"target_name": "receptor1"},
        ligand_kwargs={"compound_name": "ligand1"},
    )
    pc = PreppedComplex.from_complex(comp)
    assert (
        pc.target.target_hash
        == "843587eb7f589836d67da772b11584da4fa02fba63d6d3f3062e98c177306abb"
    )
    assert (
        pc.hash
        == "843587eb7f589836d67da772b11584da4fa02fba63d6d3f3062e98c177306abb+JZJCSVMJFIAMQB-DLYUOGNHNA-N"
    )
