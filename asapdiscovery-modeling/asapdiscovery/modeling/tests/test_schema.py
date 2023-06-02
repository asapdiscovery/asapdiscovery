from hashlib import sha256

import pydantic
import pytest
from asapdiscovery.data.openeye import oechem
from asapdiscovery.data.testing.test_resources import fetch_test_file
from asapdiscovery.modeling.modeling import save_design_unit
from asapdiscovery.modeling.schema import MoleculeFilter, PreppedTarget


@pytest.mark.parametrize(
    "components",
    [["ligand"], ["protein", "ligand"], ["protein", "ligand", "water"]],
)
def test_molecule_filter_component_success(components):
    MoleculeFilter(components_to_keep=components)


@pytest.mark.parametrize(
    "components",
    [["prot"], ["basket", "ligand"], "help"],
)
def test_molecule_filter_component_failure(components):
    with pytest.raises(ValueError):
        MoleculeFilter(components_to_keep=components)


@pytest.mark.parametrize(
    "ligand_chain",
    [list("A")],
)
def test_molecule_filter_ligand_chain_failure(ligand_chain):
    with pytest.raises((ValueError, pydantic.ValidationError)):
        MoleculeFilter(ligand_chain=ligand_chain)


def test_prepped_target():
    du_path = fetch_test_file("Mpro-P2660_0A_bound-prepped_receptor_0.oedu")
    du = oechem.OEDesignUnit()
    oechem.OEReadDesignUnit(str(du_path), du)
    prepped_target = PreppedTarget(
        output_name="test",
        molecule_filter=MoleculeFilter(components_to_keep=["protein", "ligand"]),
        output_dir="test_prep",
    )
    assert prepped_target.prepped is False

    prepped_target.set_prepped()
    assert prepped_target.prepped

    prepped_target = save_design_unit(du, prepped_target)
