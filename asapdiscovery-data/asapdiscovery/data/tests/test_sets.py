from itertools import product

from asapdiscovery.data.schema.pairs import CompoundStructurePair
from asapdiscovery.data.schema.sets import CompoundMultiStructure


def test_multi_structure_from_pairs(ligands_from_complexes, complexes):
    pairs = [
        CompoundStructurePair(ligand=ligand, complex=complex)
        for ligand, complex in product(ligands_from_complexes, complexes)
    ]
    assert len(pairs) == 40
    multi_structures = CompoundMultiStructure.from_pairs(pairs)

    assert len(multi_structures) == 4

    # check equality and inequality methods
    multi_structures_2 = CompoundMultiStructure.from_pairs(pairs)
    assert multi_structures[0] == multi_structures_2[0]
    assert multi_structures[0] != multi_structures_2[1]

    assert (
        multi_structures[0].unique_name
        == "test-KTTFDHLIVPVQIE-UHFFFAOYNA-N_de2fc89d89986c55d83706d99acce170a34caebb3cc1de6d2a0780b9c1a9fd7f"
    )
    assert (
        multi_structures[1].unique_name
        == "test-AVCQLYXAEKNILW-UHFFFAOYNA-N_de2fc89d89986c55d83706d99acce170a34caebb3cc1de6d2a0780b9c1a9fd7f"
    )

    # the ligands should be different, but the complexes should be the same
    assert multi_structures[0].ligand != multi_structures[1].ligand
    assert multi_structures[0].complexes == multi_structures[1].complexes
