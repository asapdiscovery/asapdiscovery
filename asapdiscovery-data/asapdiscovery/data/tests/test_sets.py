from itertools import product

from asapdiscovery.data.schema_v2.pairs import CompoundStructurePair
from asapdiscovery.data.schema_v2.sets import CompoundMultiStructure


def test_multi_structure_from_pairs(ligands, complexes):
    # I could use the pairwise selector but that makes it an integration test, right?
    pairs = [
        CompoundStructurePair(ligand=ligand, complex=complex)
        for ligand, complex in product(ligands, complexes)
    ]
    assert len(pairs) == 40
    multi_structures = CompoundMultiStructure.from_pairs(pairs)

    assert len(multi_structures) == 4

    # check equality and inequality methods
    multi_structures_2 = CompoundMultiStructure.from_pairs(pairs)
    assert multi_structures[0] == multi_structures_2[0]
    assert multi_structures[0] != multi_structures_2[1]
