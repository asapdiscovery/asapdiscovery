import pytest
from asapdiscovery.data.backend.openeye import get_SD_data, oechem, oemol_to_inchikey
from asapdiscovery.data.schema.ligand import Ligand
from asapdiscovery.docking.schema.pose_generation import (
    OpenEyeConstrainedPoseGenerator,
    RDKitConstrainedPoseGenerator,
)
from rdkit import Chem


def test_openeye_prov():
    """Make sure the software versions are correctly captured."""

    pose_generator = OpenEyeConstrainedPoseGenerator()
    provenance = pose_generator.provenance()
    assert "oechem" in provenance
    assert "oeomega" in provenance
    assert "oedocking" in provenance
    assert "oeff" in provenance


def test_rdkit_prov():
    """Make sure the software versions are correctly captured."""

    pose_gen = RDKitConstrainedPoseGenerator()
    provenance = pose_gen.provenance()
    assert "oechem" in provenance
    assert "oeff" in provenance
    assert "oedocking" in provenance
    assert "rdkit" in provenance
    assert "openff.toolkit" in provenance


def test_openeye_generate_core_fragment():
    """Make sure a core fragment can be extracted from a reference ligand."""
    asprin = Ligand.from_smiles("O=C(C)Oc1ccccc1C(=O)O", compound_name="asprin")
    core_smarts = "OC(=O)C"
    pose_generator = OpenEyeConstrainedPoseGenerator()
    core_mol = pose_generator._generate_core_fragment(
        reference_ligand=asprin.to_oemol(), core_smarts=core_smarts
    )
    assert oemol_to_inchikey(core_mol) == "UXTFKIJKRJJXNV-UHFFFAOYSA-N"


def test_openeye_core_fragment_not_possible():
    """Make sure an error is raised if we can not extract a subset of the mol based on the smarts."""
    asprin = Ligand.from_smiles("O=C(C)Oc1ccccc1C(=O)O", compound_name="asprin")
    core_smarts = "c1ccccc1-c2ccccc2"  # look for biphenyl substructure
    pose_generator = OpenEyeConstrainedPoseGenerator()
    with pytest.raises(RuntimeError, match="A core fragment could not be extracted "):
        _ = pose_generator._generate_core_fragment(
            reference_ligand=asprin.to_oemol(), core_smarts=core_smarts
        )


def test_generate_omega():
    """Make sure omega is correctly made from the settings."""
    asprin = Ligand.from_smiles("O=C(C)Oc1ccccc1C(=O)O", compound_name="asprin")
    pose_generator = OpenEyeConstrainedPoseGenerator()
    omega = pose_generator._generate_omega_instance(
        core_fragment=asprin.to_oemol(), use_mcs=True
    )
    assert omega.GetEnergyWindow() == pose_generator.energy_window
    assert omega.GetMaxConfs() == pose_generator.max_confs


def test_prune_clashes(mol_with_constrained_confs, mac1_complex):
    """Make sure clashes are correctly identified and pruned as required.
    The input was generated using the pose generator and writing out all conformations before pruning.
    """
    pose_generator = OpenEyeConstrainedPoseGenerator()
    # make sure all conformers are loaded
    assert 187 == mol_with_constrained_confs.NumConfs()
    oedu_receptor = mac1_complex.target.to_oedu()
    oe_receptor = oechem.OEGraphMol()
    oedu_receptor.GetProtein(oe_receptor)

    # edit the ligand conformers in place
    pose_generator._prune_clashes(
        receptor=oe_receptor, ligands=[mol_with_constrained_confs]
    )
    assert mol_with_constrained_confs.NumConfs() == 93


@pytest.mark.parametrize(
    "chemgauss, best_score",
    [
        pytest.param("Chemgauss4", -9.747581481933594, id="Chemgauss4"),
        pytest.param("Chemgauss3", -74.73448944091797, id="Chemgauss3"),
    ],
)
def test_select_best_chemgauss(
    chemgauss, best_score, mol_with_constrained_confs, mac1_complex
):
    """Make sure the conformers are correctly scored and the scoring function is changed when requested."""
    pose_generator = OpenEyeConstrainedPoseGenerator(selector=chemgauss)
    # make sure all conformers are present
    assert 187 == mol_with_constrained_confs.NumConfs()

    oedu_receptor = mac1_complex.target.to_oedu()
    oe_receptor = oechem.OEGraphMol()
    oedu_receptor.GetProtein(oe_receptor)

    single_conf_ligands = pose_generator._select_best_pose(
        receptor=mac1_complex.target.to_oedu(), ligands=[mol_with_constrained_confs]
    )
    assert isinstance(
        single_conf_ligands[0], oechem.OEGraphMol
    )  # checks its a single conf mol
    assert mol_with_constrained_confs.GetCoords() != single_conf_ligands[0].GetCoords()
    assert float(
        get_SD_data(single_conf_ligands[0])[f"{chemgauss}_score"][0]
    ) == pytest.approx(best_score)


@pytest.mark.parametrize(
    "forcefield, ff_energy",
    [
        pytest.param("MMFF", 43.42778156043702, id="MMFF"),
        pytest.param("Sage", 51.18920813475846, id="Sage"),
        pytest.param("Parsley", 128.38592742407758, id="Parsley"),
    ],
)
def test_select_by_energy(forcefield, ff_energy, mol_with_constrained_confs):
    """Test sorting the conformers by energy."""
    pose_generator = OpenEyeConstrainedPoseGenerator(backup_score=forcefield)
    # make sure all conformers are present
    assert 187 == mol_with_constrained_confs.NumConfs()

    # select best pose by energy
    best_pose = pose_generator._select_by_energy(ligand=mol_with_constrained_confs)
    assert mol_with_constrained_confs.GetActive().GetCoords() != best_pose.GetCoords()
    assert float(get_SD_data(best_pose)[f"{forcefield}_energy"][0]) == pytest.approx(
        ff_energy
    )


def test_omega_fail_codes(mac1_complex):
    """Make sure the omega failure code is captured when a conformer can not be made."""

    pose_generator = OpenEyeConstrainedPoseGenerator()
    target_ligand = Ligand.from_smiles(
        "CCNC(=O)c1cc2c([nH]1)ncnc2N[C@@H](c3ccc4c(c3)S(=O)(=O)CCC4)C5CC5",
        compound_name="omega-error",
    )
    posed_ligands = pose_generator.generate_poses(
        prepared_complex=mac1_complex,
        ligands=[target_ligand],
        core_smarts="CC(Nc1ncnc2c1cc[nH]2)c3cc(S(=O)(CCC4)=O)c4cc3",
    )
    assert len(posed_ligands.posed_ligands) == 0
    # we should have one failure as the smarts does not match
    assert len(posed_ligands.failed_ligands) == 1
    assert (
        posed_ligands.failed_ligands[0].tags["omega_return_code"]
        == "No fixed fragment found"
    )


@pytest.mark.parametrize(
    "generator",
    [
        pytest.param(OpenEyeConstrainedPoseGenerator, id="Openeye"),
        pytest.param(RDKitConstrainedPoseGenerator, id="RDKit"),
    ],
)
@pytest.mark.parametrize(
    "core_smarts",
    [
        pytest.param("CC1=CC2=C(CCCS2(=O)=O)C=C1", id="Core provided"),
        pytest.param(None, id="No core"),
    ],
)
def test_mcs_generate(mac1_complex, generator, core_smarts):
    """Make sure we can generate a conformer using the mcs when we do not pass a core smarts"""

    pose_generator = generator()
    target_ligand = Ligand.from_smiles(
        "CCNC(=O)c1cc2c([nH]1)ncnc2N[C@@H](c3ccc4c(c3)S(=O)(=O)CCC4)C5CC5",
        compound_name="omega-error",
    )
    posed_ligands = pose_generator.generate_poses(
        prepared_complex=mac1_complex, ligands=[target_ligand], core_smarts=core_smarts
    )
    assert len(posed_ligands.posed_ligands) == 1
    # we should have no fails
    assert len(posed_ligands.failed_ligands) == 0


def test_coord_transfer_fail():
    """Make sure an error is raised if we try and transfer the coords with no matching substructure."""
    asprin = Chem.MolFromSmiles("O=C(C)Oc1ccccc1C(=O)O")
    biphenyl = Chem.MolFromSmiles("c1ccccc1-c2ccccc2")  # look for biphenyl substructure

    pose_generator = RDKitConstrainedPoseGenerator()
    with pytest.raises(RuntimeError):
        pose_generator._transfer_coordinates(
            reference_ligand=asprin, template_ligand=biphenyl
        )
