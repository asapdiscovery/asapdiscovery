from pathlib import Path

import pydantic
import pytest
from asapdiscovery.data.openeye import oechem
from asapdiscovery.data.testing.test_resources import fetch_test_file
from asapdiscovery.modeling.schema import MoleculeFilter


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
