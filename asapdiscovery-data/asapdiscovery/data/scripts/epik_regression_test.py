"""A regression test for the Epik expander as we can not test this using the github CI"""

from asapdiscovery.data.operators.state_expanders.protomer_expander import EpikExpander
from asapdiscovery.data.schema.ligand import Ligand


def main():
    """
    Run Epik on two input molecules one which should return x microstates and the second which should return itself only
    and confirm the output.
    """

    input_ligands = [
        Ligand.from_smiles("c1[nH]c2c(=O)[nH]c(nc2n1)N", compound_name="expandable"),
        Ligand.from_smiles("CC", compound_name="ethane"),
    ]

    expander = EpikExpander()

    expanded_ligands = expander.expand(ligands=input_ligands)

    for ligand in expanded_ligands:
        # make sure the Epik tags have been set
        assert "r_epik_State_Penalty" in ligand.tags
    assert len(expanded_ligands) == 3


if __name__ == "__main__":
    main()
