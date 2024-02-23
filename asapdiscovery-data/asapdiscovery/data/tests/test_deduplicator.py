from asapdiscovery.data.operators.deduplicator import LigandDeDuplicator


def test_deduplication_all_unique(ligands):
    ld = LigandDeDuplicator()
    assert len(ld.deduplicate(ligands)) == 4


def test_deduplication_duplicates(ligands):
    ld = LigandDeDuplicator()
    ligands = ligands * 2
    assert len(ld.deduplicate(ligands)) == 4
