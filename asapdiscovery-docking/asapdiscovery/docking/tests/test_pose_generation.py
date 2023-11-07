import pytest

from asapdiscovery.data.openeye import oemol_to_inchikey
from asapdiscovery.data.schema_v2.ligand import Ligand
from asapdiscovery.docking.schema.pose_generation import OpenEyeConstrainedPoseGenerator


def test_openeye_prov():
    """Make sure the software versions are correctly captured."""

    pose_generator = OpenEyeConstrainedPoseGenerator()
    provenance = pose_generator.provenance()
    assert "oechem" in provenance
    assert "oeomega" in provenance
    assert "oedocking" in provenance
    assert "oeff" in provenance


def test_openeye_generate_core_fragment():
    """Make sure a core fragment can be extracted from a reference ligand."""
    asprin = Ligand.from_smiles("O=C(C)Oc1ccccc1C(=O)O", compound_name="asprin")
    core_smarts = "OC(=O)C"
    pose_generator = OpenEyeConstrainedPoseGenerator()
    core_mol = pose_generator._generate_core_fragment(
        reference_ligand=asprin, core_smarts=core_smarts
    )
    assert oemol_to_inchikey(core_mol) == "UXTFKIJKRJJXNV-UHFFFAOYSA-N"


def test_openeye_core_fragment_not_possible():
    """Make sure an error is raised if we can not extract a subset of the mol based on the smarts."""
    asprin = Ligand.from_smiles("O=C(C)Oc1ccccc1C(=O)O", compound_name="asprin")
    core_smarts = "c1ccccc1-c2ccccc2"  # look for biphenyl substructure
    pose_generator = OpenEyeConstrainedPoseGenerator()
    with pytest.raises(RuntimeError, match="A core fragment could not be extracted "):
        _ = pose_generator._generate_core_fragment(
            reference_ligand=asprin, core_smarts=core_smarts
        )


def test_generate_omega():
    """Make sure omega is correctly made from the settings."""
    asprin = Ligand.from_smiles("O=C(C)Oc1ccccc1C(=O)O", compound_name="asprin")
    pose_generator = OpenEyeConstrainedPoseGenerator()
    omega = pose_generator._generate_omega_instance(core_fragment=asprin.to_oemol())
    assert omega.GetEnergyWindow() == pose_generator.energy_window
    assert omega.GetMaxConfs() == pose_generator.max_confs
