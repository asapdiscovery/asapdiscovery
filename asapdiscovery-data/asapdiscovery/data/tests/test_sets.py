from asapdiscovery.data.schema_v2.pairs import CompoundStructurePair
from asapdiscovery.data.schema_v2.sets import CompoundMultiStructure


def test_multi_structure_from_pairs(ligands, complexes):
    pairs = [
        CompoundStructurePair(ligand=ligand, complex=complex)
        for ligand, complex in enumerate(ligands, complexes)
    ]
    assert len(pairs) == 40
    multi_structures = CompoundMultiStructure.from_pairs(pairs)

    assert len(multi_structures) == 10
