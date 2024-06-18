import pytest
from asapdiscovery.alchemy.schema.prep_workflow import (
    AlchemyPrepWorkflow,
    OpenEyeConstrainedPoseGenerator,
    RDKitConstrainedPoseGenerator,
)
from asapdiscovery.data.schema.ligand import Ligand


@pytest.mark.parametrize(
    "strict_stereo, core_smarts, failed",
    [
        pytest.param(True, None, True, id="Strict-no-smarts"),
        pytest.param(
            True,
            "C(Nc1ncnc2c1cc[nH]2)c3cc(S(=O)(CCC4)=O)c4cc3",
            False,
            id="Strict-smarts",
        ),
        pytest.param(False, None, False, id="Lax-stereo"),
        pytest.param(
            False,
            "CC(Nc1ncnc2c1cc[nH]2)c3cc(S(=O)(CCC4)=O)c4cc3",
            False,
            id="Lax-stereo-smarts",
        ),
    ],
)
def test_prep_workflow(strict_stereo, core_smarts, failed, mac1_complex):
    """Make sure the full prep workflow can be run and stereo issues can be filtered."""

    # we do not have access to epik in testing so skip, use openeye as it's faster
    workflow = AlchemyPrepWorkflow(
        charge_expander=None,
        strict_stereo=strict_stereo,
        core_smarts=core_smarts,
        pose_generator=OpenEyeConstrainedPoseGenerator(),
        charge_method=None,
    )

    alchemy_dataset = workflow.create_alchemy_dataset(
        dataset_name="mac1-testing-dataset",
        ligands=[
            Ligand.from_smiles(
                "Cc1c(cn(n1)C)c2cc3c([nH]2)ncnc3N[C@@H](c4ccc5c(c4)S(=O)(=O)CCC5)C(C)C",
                compound_name="stereo_mol",
            )
        ],
        reference_complex=mac1_complex,
    )

    assert len(alchemy_dataset.input_ligands) == 1
    assert set(alchemy_dataset.provenance.keys()) == {
        "OpenEyeConstrainedPoseGenerator",
        "StereoExpander",
    }
    assert alchemy_dataset.provenance["StereoExpander"]["expander"] == {
        "expander_type": "StereoExpander",
        "stereo_expand_defined": False,
    }
    if failed:
        assert len(alchemy_dataset.posed_ligands) == 0
        assert "InconsistentStereo" in alchemy_dataset.failed_ligands
        assert len(alchemy_dataset.failed_ligands["InconsistentStereo"]) == 1
        # make sure the excluded molecule is our input
        assert (
            alchemy_dataset.failed_ligands["InconsistentStereo"][0].compound_name
            == "stereo_mol"
        )
    else:
        assert alchemy_dataset.failed_ligands is None
        assert len(alchemy_dataset.posed_ligands) == 1
        assert alchemy_dataset.posed_ligands[0].compound_name == "stereo_mol"


def test_prep_mcs_sort():
    """Test sorting ligands by MCS match to some reference."""
    ref_molecule = Ligand.from_smiles("c1ccccc1-c2ccccc2", compound_name="biphenyl")

    exp_ligands = [
        Ligand.from_smiles("c1ccccc1", compound_name="benzene"),
        Ligand.from_smiles("CCO", compound_name="ethanol"),
        Ligand.from_smiles("c1ccccc1-c2ccncc2", compound_name="biphenyl-pyridine"),
    ]

    sorted_ligands = AlchemyPrepWorkflow._sort_similar_molecules(
        reference_ligand=ref_molecule, experimental_ligands=exp_ligands
    )

    assert sorted_ligands[0].compound_name == "biphenyl-pyridine"
    assert sorted_ligands[1].compound_name == "benzene"
    assert sorted_ligands[2].compound_name == "ethanol"


def test_prep_remove_fails():
    """Test removing a list of target ligands from an input list."""

    ligands = [
        Ligand.from_smiles("CCO", compound_name="ethanol"),
        Ligand.from_smiles("CC", compound_name="ethane"),
    ]
    fails = ligands[1:]
    filtered_ligands = AlchemyPrepWorkflow._remove_fails(
        posed_ligands=ligands, stereo_issue_ligands=fails
    )
    assert len(filtered_ligands) == 1
    assert filtered_ligands[0].compound_name == "ethanol"


def test_prep_deduplicate():
    """Test deduplicating two lists of ligands and marking the overlapping ligands."""

    ligands = [
        Ligand.from_smiles("CCO", compound_name="ethanol"),
        Ligand.from_smiles("CC", compound_name="ethane"),
    ]
    experimental_ligands = [
        Ligand.from_smiles(
            "CCO", compound_name="drug-1", **{"cdd_protocol": "my-protocol"}
        ),
        Ligand.from_smiles(
            "C", compound_name="drug-2", **{"cdd_protocol": "my-protocol"}
        ),
    ]
    filtered_ligands = AlchemyPrepWorkflow._deduplicate_experimental_ligands(
        posed_ligands=ligands, experimental_ligands=experimental_ligands
    )
    assert len(filtered_ligands) == 1
    assert filtered_ligands[0].compound_name == "drug-2"
    # make sure the overlapping molecules have been tagged
    assert ligands[0].compound_name == "ethanol"
    assert ligands[0].tags["cdd_protocol"] == "my-protocol"
    assert ligands[0].tags["experimental"] == "True"


def test_prep_workflow_ref_ligands(mac1_complex):
    """Test adding poses for experimental ligands while running the normal workflow."""
    import rich

    console = rich.get_console()

    # no access to epik so skip
    workflow = AlchemyPrepWorkflow(
        charge_expander=None,
        strict_stereo=True,
        core_smarts=None,
        # use small number of confs to keep the test fast
        pose_generator=RDKitConstrainedPoseGenerator(max_confs=10),
        # turn off charging for speed
        charge_method=None,
    )

    experimental_data = {"cdd_protocol": "my-protocol", "experimental": "True"}
    with console.capture() as capture:
        alchemy_dataset = workflow.create_alchemy_dataset(
            dataset_name="mac1-testing-dataset",
            ligands=[
                Ligand.from_smiles(
                    "Cc1c(cn(n1)C)c2cc3c([nH]2)ncnc3N[C@H](c4ccc5c(c4)S(=O)(=O)CCC5)C(C)C",
                    compound_name="stereo_mol",
                )
            ],
            reference_complex=mac1_complex,
            # add two experimental ligands one which overlaps and one unique
            reference_ligands=[
                Ligand.from_smiles(
                    "CC(C)[C@H](Nc1ccccc1)c3ccc2CCCS(=O)(=O)c2c3",
                    compound_name="ref_mol",
                    **experimental_data
                ),
                Ligand.from_smiles(
                    "Cc1c(cn(n1)C)c2cc3c([nH]2)ncnc3N[C@H](c4ccc5c(c4)S(=O)(=O)CCC5)C(C)C",
                    compound_name="ref_stereo_mol",
                    **experimental_data
                ),
            ],
        )
    assert len(alchemy_dataset.input_ligands) == 1
    # we should have the input molecule and the experimental molecule
    assert len(alchemy_dataset.posed_ligands) == 2
    # make sure the overlapping pose molecules name was not changed
    assert alchemy_dataset.posed_ligands[0].compound_name == "stereo_mol"
    # make sure the experimental ligand is unchanged
    assert alchemy_dataset.posed_ligands[1].compound_name == "ref_mol"
    # make sure they both have experimental tags
    for mol in alchemy_dataset.posed_ligands:
        for key, value in experimental_data.items():
            assert mol.tags[key] == value

    assert (
        "Injected ligand: ref_mol; SMILES: CC(C)[C@@H](c1ccc2c(c1)S(=O)(=O)CCC2)Nc3ccccc3"
        in capture.get()
    )


def test_prep_with_charges(mac1_complex):
    """Test running the prep workflow and generating charges"""

    # no access to epik so skip
    workflow = AlchemyPrepWorkflow(
        charge_expander=None,
        strict_stereo=True,
        core_smarts=None,
        # use small number of confs to keep the test fast
        pose_generator=RDKitConstrainedPoseGenerator(max_confs=10),
    )

    alchemy_dataset = workflow.create_alchemy_dataset(
        dataset_name="mac1-testing-dataset",
        ligands=[
            Ligand.from_smiles(
                "Cc1c(cn(n1)C)c2cc3c([nH]2)ncnc3N[C@H](c4ccc5c(c4)S(=O)(=O)CCC5)C(C)C",
                compound_name="stereo_mol",
            )
        ],
        reference_complex=mac1_complex,
    )
    # check we have the expected number of outputs
    assert len(alchemy_dataset.input_ligands) == 1
    assert len(alchemy_dataset.posed_ligands) == 1
    # make sure charges were generated, name must be this to be found by openfe and openff
    assert "atom.dprop.PartialCharge" in alchemy_dataset.posed_ligands[0].tags

    # mock the BFE workflow passing the charges to openfe  then openff and make sure they match
    openfe_mol = alchemy_dataset.posed_ligands[0].to_openfe()
    off_mol = openfe_mol.to_openff()
    assert off_mol.partial_charges is not None
    # make sure the charges are consistent
    for i, charge in enumerate(
        alchemy_dataset.posed_ligands[0].tags["atom.dprop.PartialCharge"].split(" ")
    ):
        assert float(charge) == off_mol.partial_charges[i].m

    # make sure the method was stamped on the molecule
    assert (
        alchemy_dataset.posed_ligands[0].charge_provenance.dict(exclude={"type"})
        == workflow.charge_method.provenance()
    )
