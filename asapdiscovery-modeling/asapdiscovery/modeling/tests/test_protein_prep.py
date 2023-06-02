from pathlib import Path

import pytest
import yaml
from asapdiscovery.data.openeye import (
    load_openeye_cif1,
    load_openeye_pdb,
    oechem,
    save_openeye_pdb,
)
from asapdiscovery.data.schema import CrystalCompoundData, CrystalCompoundDataset
from asapdiscovery.data.testing.test_resources import fetch_test_file
from asapdiscovery.data.utils import seqres_to_res_list
from asapdiscovery.modeling.modeling import (
    add_seqres_to_openeye_protein,
    mutate_residues,
    split_openeye_design_unit,
    split_openeye_mol,
    spruce_protein,
    superpose_molecule,
)
from asapdiscovery.modeling.schema import MoleculeFilter, PreppedTarget, PreppedTargets


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
    if not type(local_path) == str:
        return tmp_path_factory.mktemp("test_prep")
    else:
        local_path = Path(local_path)
        local_path.mkdir(exist_ok=True)
        assert local_path.exists()
        return local_path


@pytest.fixture
def sars_xtal(sars):
    return CrystalCompoundData(
        str_fn=str(sars),
    )


@pytest.fixture
def sars_target(sars_xtal):
    return PreppedTarget(
        source=sars_xtal,
        active_site_chain="A",
        output_name=Path(sars_xtal.str_fn).stem,
        molecule_filter=MoleculeFilter(
            components_to_keep=["protein", "ligand"], ligand_chain="A"
        ),
    )


@pytest.fixture
def mers_xtal(mers):
    return CrystalCompoundData(
        str_fn=str(mers),
    )


@pytest.fixture
def mers_target(mers_xtal):
    return PreppedTarget(
        source=mers_xtal,
        active_site_chain="A",
        output_name=Path(mers_xtal.str_fn).stem,
        active_site="HIS:41: :A:0: ",
        molecule_filter=MoleculeFilter(components_to_keep=["protein"]),
    )


@pytest.fixture
def target_dataset(sars_target, mers_target):
    target_dataset = PreppedTargets.from_list([sars_target, mers_target])
    print(target_dataset)
    return target_dataset


class TestCrystalCompoundDataset:
    def test_dataset_creation(self, target_dataset, sars_target, mers_target):
        assert len(target_dataset.iterable) == 2
        assert target_dataset.iterable[0].source.str_fn == str(
            sars_target.source.str_fn
        )
        assert target_dataset.iterable[1].source.str_fn == str(
            mers_target.source.str_fn
        )

    def test_dataset_saving(
        self, target_dataset, prepped_files, csv_name="to_prep.csv"
    ):
        to_prep_csv = prepped_files / csv_name
        target_dataset.to_csv(to_prep_csv)
        assert to_prep_csv.exists()
        assert to_prep_csv.is_file()

    def test_dataset_loading(
        self, target_dataset, prepped_files, csv_name="to_prep.csv"
    ):
        dataset = PreppedTargets.from_csv(prepped_files / csv_name)
        assert dataset == target_dataset

        dataset.iterable[0].active_site_chain = "B"
        assert dataset != target_dataset

    def test_dataset_pickle(
        self, target_dataset, prepped_files, pkl_name="to_prep.pkl"
    ):
        pkl_file = prepped_files / pkl_name
        target_dataset.to_pkl(pkl_file)
        assert pkl_file.exists()
        assert pkl_file.is_file()

        loaded_dataset = PreppedTargets.from_pkl(pkl_file)

        assert loaded_dataset == target_dataset

    def test_dataset_json(
        self, target_dataset, prepped_files, json_name="to_prep.json"
    ):
        json_file = prepped_files / json_name
        target_dataset.to_json(json_file)
        assert json_file.exists()
        assert json_file.is_file()

        loaded_dataset = PreppedTargets.from_json(json_file)

        assert loaded_dataset == target_dataset


class TestProteinPrep:
    def test_sars_protein_prep(
        self, sars_target, ref, prepped_files, loop_db, ref_chain="A"
    ):
        xtal = sars_target

        # Load structure
        prot = load_openeye_pdb(xtal.str_fn)
        ref = load_openeye_pdb(str(ref))

        assert type(prot) == oechem.OEGraphMol

        # Get protein and ligand
        prot = split_openeye_mol(
            prot,
            MoleculeFilter(
                components_to_keep=["protein", "ligand"],
                protein_chains=["A", "B"],
                ligand_chain="B",
            ),
        )
        save_openeye_pdb(prot, prepped_files / f"{xtal.output_name}_split.pdb")

        aligned, rmsd = superpose_molecule(ref, prot, ref_chain, "A")

        save_openeye_pdb(aligned, prepped_files / f"{xtal.output_name}_align.pdb")

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
        from asapdiscovery.data.openeye import save_openeye_sdf
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

    def test_mers_protein_prep(
        self, mers_xtal, ref, prepped_files, loop_db, ref_chain="A"
    ):
        xtal = mers_xtal
        # Load structure
        prot = load_openeye_cif1(xtal.str_fn)
        ref = load_openeye_pdb(str(ref))

        assert type(prot) == oechem.OEGraphMol

        # Get only protein
        prot = split_openeye_mol(
            prot,
            MoleculeFilter(
                components_to_keep=["protein", "ligand"],
                protein_chains=["A", "B"],
                ligand_chain="B",
            ),
        )
        save_openeye_pdb(prot, prepped_files / f"{xtal.output_name}_split.pdb")

        prot, rmsd = superpose_molecule(ref, prot, ref_chain, "A")

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
