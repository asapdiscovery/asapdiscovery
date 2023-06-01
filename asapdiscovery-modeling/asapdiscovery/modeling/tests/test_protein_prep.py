import pytest
from asapdiscovery.data.testing.test_resources import fetch_test_file
from asapdiscovery.data.schema import CrystalCompoundData, CrystalCompoundDataset
import yaml
from asapdiscovery.data.utils import seqres_to_res_list
from asapdiscovery.modeling.modeling import (
    remove_extra_ligands,
    align_receptor,
    spruce_protein,
    mutate_residues,
    add_seqres_to_openeye_protein,
)
from asapdiscovery.data.openeye import (
    oechem,
    load_openeye_pdb,
    save_openeye_pdb,
    load_openeye_cif1,
)
from pathlib import Path


@pytest.fixture
def sars():
    return fetch_test_file("Mpro-P2660_0A_bound.pdb")


@pytest.fixture
def mers():
    return fetch_test_file("rcsb_8czv-assembly1.cif")


@pytest.fixture
def ref():
    return fetch_test_file("reference.pdb")


@pytest.fixture
def loop_db():
    return fetch_test_file("fragalysis-mpro_spruce.loop_db")


# This needs to have a scope of session so that a new tmp file is not created for each test
@pytest.fixture(scope="session")
def prepped_files(tmp_path_factory, local_path):
    if not type(local_path) == Path:
        return tmp_path_factory.mktemp("test_prep")
    else:
        local_path.mkdir(exist_ok=True)
        return local_path


@pytest.fixture
def sars_xtal(sars):
    return CrystalCompoundData(
        str_fn=str(sars),
        active_site_chain="A",
        output_name=Path(sars).stem,
        lig_chain="A",
    )


@pytest.fixture
def mers_xtal(mers):
    return CrystalCompoundData(
        str_fn=str(mers),
        active_site_chain="A",
        output_name=Path(mers).stem,
        active_site="HIS:41: :A:0: ",
        lig_chain="B",
    )


@pytest.fixture
def xtal_dataset(sars_xtal, mers_xtal):
    xtal_dataset = CrystalCompoundDataset(structures=[sars_xtal, mers_xtal])
    return xtal_dataset


class TestCrystalCompoundDataset:
    def test_dataset_creation(self, xtal_dataset, sars_xtal, mers_xtal):
        assert len(xtal_dataset.structures) == 2
        assert xtal_dataset.structures[0].str_fn == str(sars_xtal.str_fn)
        assert xtal_dataset.structures[1].str_fn == str(mers_xtal.str_fn)

    def test_dataset_saving(self, xtal_dataset, prepped_files, csv_name="to_prep.csv"):
        to_prep_csv = prepped_files / csv_name
        xtal_dataset.to_csv(to_prep_csv)
        assert to_prep_csv.exists()
        assert to_prep_csv.is_file()

    def test_dataset_loading(self, xtal_dataset, prepped_files, csv_name="to_prep.csv"):
        dataset = CrystalCompoundDataset()
        dataset.from_csv(prepped_files / csv_name)
        assert dataset == xtal_dataset

        dataset.structures[0].active_site_chain = "B"
        assert dataset != xtal_dataset


class TestProteinPrep:
    def test_sars_protein_prep(
        self, sars_xtal, ref, prepped_files, loop_db, ref_chain="A"
    ):
        xtal = sars_xtal

        # Load structure
        prot = load_openeye_pdb(xtal.str_fn)

        assert type(prot) == oechem.OEGraphMol

        prot = remove_extra_ligands(prot, lig_chain=xtal.lig_chain)

        # TODO add a test to confirm only a ligand in chain A is there

        # Align to reference
        prot = align_receptor(
            prot,
            dimer=True,
            ref_prot=str(ref),
            split_initial_complex=False,
            split_ref=True,
            ref_chain=ref_chain,
            mobile_chain=xtal.active_site_chain,
            keep_water=False,
        )

        # TODO add test to make sure the active site is aligned

        save_openeye_pdb(prot, prepped_files / f"{xtal.output_name}_align.pdb")

        # Mutate Residues
        seqres_yaml = fetch_test_file("mpro_sars2_seqres.yaml")

        with open(seqres_yaml) as f:
            seqres_dict = yaml.safe_load(f)
        seqres = seqres_dict["SEQRES"]

        res_list = seqres_to_res_list(seqres)

        prot = mutate_residues(prot, res_list, place_h=True)
        seqres = " ".join(res_list)

        save_openeye_pdb(prot, prepped_files / f"{xtal.output_name}_mutate.pdb")

        # Spruce Protein
        du = spruce_protein(
            initial_prot=prot,
            seqres=seqres,
            loop_db=loop_db,
            return_du=True,
            site_residue=xtal.active_site,
        )
        assert type(du) == oechem.OEDesignUnit

        du_fn = prepped_files / f"{xtal.output_name}-prepped_receptor_0.oedu"
        oechem.OEWriteDesignUnit(str(du_fn), du)

        # Serialize output!
        from asapdiscovery.data.openeye import (
            save_openeye_sdf,
            split_openeye_design_unit,
        )
        from asapdiscovery.modeling.modeling import add_seqres_to_openeye_protein

        # TODO: Use a different way of splitting the design unit
        lig, prot, complex_ = split_openeye_design_unit(du, lig_title=xtal.compound_id)
        prot = add_seqres_to_openeye_protein(prot, seqres)
        complex_ = add_seqres_to_openeye_protein(complex_, seqres)

        prot_fn = prepped_files / f"{xtal.output_name}-prepped_protein.pdb"
        save_openeye_pdb(prot, str(prot_fn))

        complex_fn = prepped_files / f"{xtal.output_name}-prepped_complex.pdb"
        save_openeye_pdb(complex_, str(complex_fn))

        lig_fn = prepped_files / f"{xtal.output_name}-prepped_ligand.sdf"
        save_openeye_sdf(lig, str(lig_fn))

        for fn in [prot_fn, complex_fn, lig_fn]:
            assert Path(fn).exists()
            assert Path(fn).is_file()

        # TODO: Add a test to make sure the ligand is in the active site
        # TODO: Add a test to make sure the seqres has been added to the protein

    # @pytest.mark.skip(reason="MERS is not ready yet")
    def test_mers_protein_prep(
        self, mers_xtal, ref, prepped_files, loop_db, ref_chain="A"
    ):
        xtal = mers_xtal
        # Load structure
        prot = load_openeye_cif1(xtal.str_fn)

        assert type(prot) == oechem.OEGraphMol

        # Align to reference
        prot = align_receptor(
            prot,
            dimer=True,
            ref_prot=str(ref),
            split_initial_complex=True,
            split_ref=True,
            ref_chain=ref_chain,
            mobile_chain=xtal.active_site_chain,
            keep_water=False,
        )

        save_openeye_pdb(prot, prepped_files / f"{xtal.output_name}_align.pdb")

        # Mutate Residues
        seqres_yaml = fetch_test_file("mpro_mers_seqres.yaml")

        with open(seqres_yaml) as f:
            seqres_dict = yaml.safe_load(f)
        seqres = seqres_dict["SEQRES"]

        res_list = seqres_to_res_list(seqres)

        prot = mutate_residues(prot, res_list, place_h=True)
        seqres = " ".join(res_list)

        save_openeye_pdb(prot, prepped_files / f"{xtal.output_name}_mutate.pdb")

        # Spruce Protein
        du = spruce_protein(
            initial_prot=prot,
            seqres=seqres,
            loop_db=loop_db,
            return_du=True,
            site_residue=xtal.active_site,
        )
        assert type(du) == oechem.OEDesignUnit

        # Serialize output!

        # TODO: Make sure this doesn't fail if there is no ligand
        du_fn = prepped_files / f"{xtal.output_name}-prepped_receptor_0.oedu"
        oechem.OEWriteDesignUnit(str(du_fn), du)

        prot = oechem.OEGraphMol()
        du.GetProtein(prot)
        prot = add_seqres_to_openeye_protein(prot, seqres)

        prot_fn = prepped_files / f"{xtal.output_name}-prepped_protein.pdb"
        save_openeye_pdb(prot, str(prot_fn))

        assert Path(prot_fn).exists()
        assert Path(prot_fn).is_file()

    @pytest.mark.parametrize(
        ("du_fn", "has_lig"),
        [
            ("Mpro-P2660_0A_bound-prepped_receptor_0.oedu", True),
            ("rcsb_8czv-assembly1-prepped_receptor_0.oedu", False),
        ],
    )
    def test_openeye_du_loading(self, du_fn, has_lig, prepped_files):
        # The purpose of this test is to make sure that the prepared design units can be used for docking

        # Load the prepared design unit
        sars_du = oechem.OEDesignUnit()
        oechem.OEReadDesignUnit(str(prepped_files / du_fn), sars_du)

        assert type(sars_du) == oechem.OEDesignUnit
        assert sars_du.HasReceptor()
        if has_lig:
            assert sars_du.HasLigand()

        # TODO: Add a test to make sure the receptor can be added to POSIT

    @pytest.mark.parametrize(
        "pdb_fn",
        [
            "rcsb_8czv-assembly1-prepped_protein.pdb",
            "Mpro-P2660_0A_bound-prepped_protein.pdb",
        ],
    )
    def test_simulation(self, pdb_fn, prepped_files):
        pdb_path = prepped_files / pdb_fn
        from asapdiscovery.simulation.testing import test_forcefield_generation

        test_forcefield_generation(str(pdb_path))
