import pytest
from asapdiscovery.data.operators.selectors.mcs_selector import (
    MCSSelector,
    RascalMCESSelector,
)
from asapdiscovery.data.operators.selectors.pairwise_selector import (
    LeaveOneOutSelector,
    LeaveSimilarOutSelector,
    PairwiseSelector,
    SelfDockingSelector,
)
from asapdiscovery.data.schema.ligand import Ligand
from asapdiscovery.data.schema.pairs import CompoundStructurePair
from asapdiscovery.docking.docking import DockingInputPair  # TODO: move to data


def test_pairwise_selector(ligands_from_complexes, complexes):
    selector = PairwiseSelector()
    pairs = selector.select(ligands_from_complexes, complexes)
    assert len(pairs) == 40


def test_leave_one_out_selector(ligands_from_complexes, complexes):
    selector = LeaveOneOutSelector()
    pairs = selector.select(ligands_from_complexes, complexes)
    assert len(pairs) == 36


def test_leave_similar_out_selector(ligands_from_complexes, complexes):
    selector = LeaveSimilarOutSelector()
    pairs = selector.select(ligands_from_complexes, complexes)
    assert len(pairs) == 36


def test_self_docking_selector(ligands_from_complexes, complexes):
    selector = SelfDockingSelector()
    pairs = selector.select(ligands_from_complexes, complexes)
    assert len(pairs) == 4


@pytest.mark.parametrize("use_dask", [True, False])
def test_pairwise_selector_prepped(ligands_from_complexes, prepped_complexes, use_dask):
    selector = PairwiseSelector()
    pairs = selector.select(
        ligands_from_complexes, prepped_complexes, use_dask=use_dask
    )
    assert len(pairs) == 8


@pytest.mark.parametrize("approximate", [True, False])
@pytest.mark.parametrize("structure_based", [True, False])
def test_mcs_selector(ligands_from_complexes, complexes, approximate, structure_based):
    selector = MCSSelector(approximate=approximate, structure_based=structure_based)
    pairs = selector.select(ligands_from_complexes, complexes, n_select=1)
    # should be 4 pairs
    assert len(pairs) == 4
    # as we matched against the exact smiles of the first 4 complex ligands_from_complexes, they should be in order
    assert pairs[0] == CompoundStructurePair(
        ligand=ligands_from_complexes[0], complex=complexes[0]
    )
    assert pairs[1] == CompoundStructurePair(
        ligand=ligands_from_complexes[1], complex=complexes[1]
    )
    assert pairs[2] == CompoundStructurePair(
        ligand=ligands_from_complexes[2], complex=complexes[2]
    )
    assert pairs[3] == CompoundStructurePair(
        ligand=ligands_from_complexes[3], complex=complexes[3]
    )


@pytest.mark.parametrize("use_dask", [True, False])
def test_rascalMCES_selector(ligands_from_complexes, complexes, use_dask):
    selector = RascalMCESSelector()
    pairs = selector.select(
        ligands_from_complexes, complexes, n_select=1, use_dask=use_dask
    )
    # should be 4 pairs
    assert len(pairs) == 4
    # as we matched against the exact smiles of the first 4 complex ligands_from_complexes, they should be in order
    assert pairs[0] == CompoundStructurePair(
        ligand=ligands_from_complexes[0], complex=complexes[0]
    )
    assert pairs[1] == CompoundStructurePair(
        ligand=ligands_from_complexes[1], complex=complexes[1]
    )
    assert pairs[2] == CompoundStructurePair(
        ligand=ligands_from_complexes[2], complex=complexes[2]
    )
    assert pairs[3] == CompoundStructurePair(
        ligand=ligands_from_complexes[3], complex=complexes[3]
    )


def test_mcs_select_prepped(ligands_from_complexes, prepped_complexes):
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


def test_mcs_selector_nselect(ligands_from_complexes, complexes):
    selector = MCSSelector()
    pairs = selector.select(ligands_from_complexes, complexes, n_select=2)
    # should be 8 pairs
    assert len(pairs) == 8
    assert (
        pairs[0].complex.ligand.smiles == "Cc1ccncc1N(C)C(=O)Cc2cccc(c2)Cl"
    )  # exact match
    assert (
        pairs[1].complex.ligand.smiles == "Cc1ccncc1NC(=O)Cc2cc(cc(c2)Cl)OC"
    )  # clearly related


def test_mcs_selector_no_match(prepped_complexes):
    lig = Ligand.from_smiles("Si", compound_name="test_no_match")
    selector = MCSSelector()
    _ = selector.select([lig], prepped_complexes, n_select=1)
