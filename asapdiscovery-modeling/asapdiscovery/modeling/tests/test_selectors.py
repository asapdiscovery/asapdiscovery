import pytest

from asapdiscovery.data.operators.selectors.mcs_selector import MCSSelector
from asapdiscovery.data.operators.selectors.pairwise_selector import PairwiseSelector
from asapdiscovery.data.schema.ligand import Ligand


@pytest.mark.parametrize("use_dask", [True, False])
def test_pairwise_selector_prepped(ligands_from_complexes, prepped_complexes, use_dask):
    selector = PairwiseSelector()
    pairs = selector.select(
        ligands_from_complexes, prepped_complexes, use_dask=use_dask
    )
    assert len(pairs) == 8


def test_mcs_select_prepped(ligands_from_complexes, prepped_complexes):
    DockingInputPair = pytest.importorskip(
        "asapdiscovery.docking.docking",
        reason="asapdiscovery-docking not installed",
    ).DockingInputPair

    selector = MCSSelector()
    pairs = selector.select(ligands_from_complexes, prepped_complexes, n_select=1)
    # should be 4 pairs
    assert len(pairs) == 4
    assert pairs[0] == DockingInputPair(
        ligand=ligands_from_complexes[0], complex=prepped_complexes[0]
    )
    assert pairs[1] == DockingInputPair(
        ligand=ligands_from_complexes[1], complex=prepped_complexes[1]
    )
    assert pairs[2] == DockingInputPair(
        ligand=ligands_from_complexes[2], complex=prepped_complexes[1]
    )
    assert pairs[3] == DockingInputPair(
        ligand=ligands_from_complexes[3], complex=prepped_complexes[0]
    )


def test_mcs_selector_no_match(prepped_complexes):
    lig = Ligand.from_smiles("Si", compound_name="test_no_match")
    selector = MCSSelector()
    _ = selector.select([lig], prepped_complexes, n_select=1)
