import pytest
from asapdiscovery.alchemy.schema.prep_workflow import (
    AlchemyPrepWorkflow,
    OpenEyeConstrainedPoseGenerator,
)
from asapdiscovery.data.schema_v2.ligand import Ligand


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
