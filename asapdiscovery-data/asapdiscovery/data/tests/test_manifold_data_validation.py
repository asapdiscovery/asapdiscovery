import pandas as pd
from asapdiscovery.data.services.postera.manifold_data_validation import (
    ManifoldAllowedTags,
    OutputTags,
    StaticTags,
    TargetTags,
)


def test_target_tags():
    assert "SARS-CoV-2-Mpro" in TargetTags.get_values()
    assert "MERS-CoV-Mpro" in TargetTags.get_values()
    assert "SARS-CoV-2-Mac1" in TargetTags.get_values()
    assert "42" not in TargetTags.get_values()


def test_output_tags():
    # select some
    assert "docking-score-POSIT" in OutputTags.get_values()
    assert "docking-confidence-POSIT" in OutputTags.get_values()
    assert "computed-FEC-pIC50" in OutputTags.get_values()
    assert "computed-FEC-uncertainty-pIC50" in OutputTags.get_values()
    assert "computed-GAT-pIC50" in OutputTags.get_values()
    assert "computed-SchNet-pIC50" in OutputTags.get_values()
    assert "42" not in OutputTags.get_values()


def test_static_tags():
    assert "SMILES" in StaticTags.get_values()
    assert "in-silico-UUID_POSTERA_API" in StaticTags.get_values()
    assert "in-silico-UUID_CCC_DB" in StaticTags.get_values()
    assert "42" not in StaticTags.get_values()


def test_allowed_tags():
    # you get the idea
    assert (
        "biochemical-activity_MERS-CoV-Mpro_computed-FEC-pIC50_msk"
        in ManifoldAllowedTags.get_values()
    )
    assert (
        "biochemical-activity_SARS-CoV-2-Mac1_computed-GAT-pIC50_msk"
        in ManifoldAllowedTags.get_values()
    )
    assert (
        "in-silico_SARS-CoV-2-Mac1_docking-confidence-POSIT_msk"
        in ManifoldAllowedTags.get_values()
    )
    assert (
        "in-silico_MERS-CoV-Mpro_docking-score-POSIT_msk"
        in ManifoldAllowedTags.get_values()
    )
    assert "in-silico_SARS-CoV-2-Mpro_md-pose_msk" in ManifoldAllowedTags.get_values()
    assert "in-silico-UUID_POSTERA_API" in ManifoldAllowedTags.get_values()
    assert "in-silico-UUID_CCC_DB" in ManifoldAllowedTags.get_values()
    assert "42" not in ManifoldAllowedTags.get_values()


def test_manifold_tags_is_in():
    assert ManifoldAllowedTags.is_in_values(
        "biochemical-activity_MERS-CoV-Mpro_computed-FEC-pIC50_msk"
    )
    assert not ManifoldAllowedTags.is_in_values("42")


def test_manifold_filter_all_in():
    assert ManifoldAllowedTags.all_in_values(
        [
            "biochemical-activity_MERS-CoV-Mpro_computed-FEC-pIC50_msk",
            "in-silico_MERS-CoV-Mpro_docking-score-POSIT_msk",
        ]
    )
    assert not ManifoldAllowedTags.all_in_values(
        [
            "biochemical-activity_MERS-CoV-Mpro_computed-FEC-pIC50_msk",
            "in-silico_MERS-CoV-Mpro_docking-score-POSIT_msk",
            "42",
        ]
    )


def test_manifold_filter_dataframe_cols():
    # make a dataframe with some columns
    df = pd.DataFrame(
        {
            "in-silico_MERS-CoV-Mpro_docking-score-POSIT_msk": [1, 2, 3],
            "in-silico_SARS-CoV-2-Mpro_docking-score-POSIT_msk": [4, 5, 6],
        }
    )
    # filter the dataframe
    ret_df = ManifoldAllowedTags.filter_dataframe_cols(df)
    assert all(df == ret_df)


def test_manifold_filter_dataframe_cols_drops():
    # make a dataframe with some columns
    df = pd.DataFrame(
        {
            "in-silico_MERS-CoV-Mpro_docking-score-POSIT_msk": [1, 2, 3],
            "in-silico_SARS-CoV-2-Mpro_docking-score-POSIT_msk": [4, 5, 6],
            "42": [7, 8, 9],
        }
    )
    # filter the dataframe
    ret_df = ManifoldAllowedTags.filter_dataframe_cols(df)
    assert all(
        df[
            [
                "in-silico_MERS-CoV-Mpro_docking-score-POSIT_msk",
                "in-silico_SARS-CoV-2-Mpro_docking-score-POSIT_msk",
            ]
        ]
        == ret_df
    )


def test_manifold_filter_dataframe_cols_doesnt_drop_allow():
    # make a dataframe with some columns
    df = pd.DataFrame(
        {
            "in-silico_MERS-CoV-Mpro_docking-score-POSIT_msk": [1, 2, 3],
            "in-silico_SARS-CoV-2-Mpro_docking-score-POSIT_msk": [4, 5, 6],
            "42": [7, 8, 9],
        }
    )
    # filter the dataframe
    ret_df = ManifoldAllowedTags.filter_dataframe_cols(df, allow=["42"])
    assert all(df == ret_df)
