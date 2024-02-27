from itertools import product

from asapdiscovery.data.schema.pairs import CompoundStructurePair
from asapdiscovery.data.schema.sets import CompoundMultiStructure


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

    assert (
        multi_structures[0].unique_name
        == "test-KTTFDHLIVPVQIE-UHFFFAOYNA-N_0bf8638845b542d72ce3375b24f0c5757fccc4e8f11b2eb65d838acc41508a16"
    )
    assert (
        multi_structures[1].unique_name
        == "test-AVCQLYXAEKNILW-UHFFFAOYNA-N_0bf8638845b542d72ce3375b24f0c5757fccc4e8f11b2eb65d838acc41508a16"
    )

    # the ligands should be different, but the complexes should be the same
    assert multi_structures[0].ligand != multi_structures[1].ligand
    assert multi_structures[0].complexes == multi_structures[1].complexes
