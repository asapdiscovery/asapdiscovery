import pandas as pd
from asapdiscovery.data.postera.manifold_data_validation import (
    ManifoldAllowedTags,
    ManifoldFilter,
    OutputTags,
    StaticTags,
    TargetTags,
)


def test_target_tags():
    assert "sars2_mpro" in TargetTags.get_values()
    assert "mers_mpro" in TargetTags.get_values()
    assert "sars2_mac1" in TargetTags.get_values()
    assert "42" not in TargetTags.get_values()


def test_output_tags():
    # select some
    assert "Docking_Score_POSIT" in OutputTags.get_values()
    assert "ML_Score_GAT_pIC50" in OutputTags.get_values()
    assert "ML_Score_Schnet_pIC50" in OutputTags.get_values()
    assert "FECs_Affinity_pIC50" in OutputTags.get_values()
    assert "FECs_Uncertainty_pIC50" in OutputTags.get_values()
    assert "42" not in OutputTags.get_values()


def test_static_tags():
    assert "SMILES" in StaticTags.get_values()
    assert "UUID_POSTERA_API" in StaticTags.get_values()
    assert "UUID_CCC_DB" in StaticTags.get_values()
    assert "42" not in StaticTags.get_values()


def test_allowed_tags():
    # you get the idea
    assert "Docking_Score_POSIT_sars2_mpro" in ManifoldAllowedTags.get_values()
    assert "Docking_Score_POSIT_mers_mpro" in ManifoldAllowedTags.get_values()
    assert "Docking_Score_POSIT_sars2_mac1" in ManifoldAllowedTags.get_values()
    assert "42" not in ManifoldAllowedTags.get_values()


def test_manifold_filter():
    assert ManifoldFilter.is_allowed_column("Docking_Score_POSIT_sars2_mpro")
    assert not ManifoldFilter.is_allowed_column("42")


def test_manifold_filter_allowed_column():
    assert ManifoldFilter.is_allowed_column("Docking_Score_POSIT_sars2_mpro")
    assert not ManifoldFilter.is_allowed_column("42")


def test_manifold_filter_all_allowed():
    assert ManifoldFilter.all_valid_columns(
        ["Docking_Score_POSIT_sars2_mpro", "Docking_Score_POSIT_mers_mpro"]
    )
    assert not ManifoldFilter.all_valid_columns(
        ["Docking_Score_POSIT_sars2_mpro", "Docking_Score_POSIT_mers_mpro", "42"]
    )


def test_manifold_filter_dataframe_cols():
    # make a dataframe with some columns
    df = pd.DataFrame(
        {
            "Docking_Score_POSIT_sars2_mpro": [1, 2, 3],
            "Docking_Score_POSIT_mers_mpro": [4, 5, 6],
        }
    )
    # filter the dataframe
    ret_df = ManifoldFilter.filter_dataframe_cols(df)
    assert all(df == ret_df)


def test_manifold_filter_dataframe_cols_drops():
    # make a dataframe with some columns
    df = pd.DataFrame(
        {
            "Docking_Score_POSIT_sars2_mpro": [1, 2, 3],
            "Docking_Score_POSIT_mers_mpro": [4, 5, 6],
            "42": [7, 8, 9],
        }
    )
    # filter the dataframe
    ret_df = ManifoldFilter.filter_dataframe_cols(df)
    assert all(
        df[["Docking_Score_POSIT_sars2_mpro", "Docking_Score_POSIT_mers_mpro"]]
        == ret_df
    )
