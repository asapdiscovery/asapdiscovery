import pytest
from asapdiscovery.docking.scorer import (
    ChemGauss4Scorer,
    E3NNScorer,
    FINTScorer,
    GATScorer,
    MetaScorer,
    SchnetScorer,
)


# parametrize over fixtures
@pytest.mark.parametrize(
    "data_fixture", ["results_simple_nolist", "complex_simple", "pdb_simple"]
)
@pytest.mark.parametrize("return_df", [True, False])
@pytest.mark.parametrize("use_dask", [True, False])
def test_chemgauss_scorer(use_dask, return_df, data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    scorer = ChemGauss4Scorer()
    scores = scorer.score([data], use_dask=use_dask, return_df=return_df)
    assert len(scores) == 1


@pytest.mark.parametrize("data_fixture", ["results_simple_nolist", "ligand", "smiles"])
@pytest.mark.parametrize("return_df", [True, False])
@pytest.mark.parametrize("use_dask", [True, False])
def test_gat_scorer(use_dask, return_df, data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    scorer = GATScorer.from_latest_by_target("SARS-CoV-2-Mpro")
    scores = scorer.score([data], use_dask=use_dask, return_df=return_df)
    assert len(scores) == 1


@pytest.mark.parametrize(
    "data_fixture", ["results_simple_nolist", "complex_simple", "pdb_simple"]
)
@pytest.mark.parametrize("return_df", [True, False])
@pytest.mark.parametrize("use_dask", [True, False])
def test_schnet_scorer(use_dask, return_df, data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    scorer = SchnetScorer.from_latest_by_target("SARS-CoV-2-Mpro")
    scores = scorer.score([data], use_dask=use_dask, return_df=return_df)
    assert len(scores) == 1


@pytest.mark.parametrize(
    "data_fixture", ["results_simple_nolist", "complex_simple", "pdb_simple"]
)
@pytest.mark.parametrize("return_df", [True, False])
@pytest.mark.parametrize("use_dask", [True, False])
def test_e3nn_scorer(use_dask, return_df, data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    scorer = E3NNScorer.from_latest_by_target("SARS-CoV-2-Mpro")
    scores = scorer.score([data], use_dask=use_dask, return_df=return_df)
    assert len(scores) == 1


@pytest.mark.parametrize("use_dask", [True, False])
def test_meta_scorer(results, use_dask):
    scorer = MetaScorer(
        scorers=[
            ChemGauss4Scorer(),
            GATScorer.from_latest_by_target("SARS-CoV-2-Mpro"),
            SchnetScorer.from_latest_by_target("SARS-CoV-2-Mpro"),
        ]
    )

    scores = scorer.score(results, use_dask=use_dask)
    assert len(scores) == 3


def test_meta_scorer_df(results_multi):
    scorer = MetaScorer(
        scorers=[
            ChemGauss4Scorer(),
            GATScorer.from_latest_by_target("SARS-CoV-2-Mpro"),
            SchnetScorer.from_latest_by_target("SARS-CoV-2-Mpro"),
            E3NNScorer.from_latest_by_target("SARS-CoV-2-Mpro"),
        ]
    )

    scores = scorer.score(results_multi, return_df=True)
    assert len(scores) == 2  # 3 scorers for each of 2 inputs


@pytest.mark.parametrize(
    "data_fixture", ["results_simple_nolist", "complex_simple", "pdb_simple"]
)
@pytest.mark.parametrize("return_df", [True, False])
@pytest.mark.parametrize("use_dask", [True, False])
def test_FINT_scorer(use_dask, return_df, data_fixture, request):
    data = request.getfixturevalue(data_fixture)
    scorer = FINTScorer(target="SARS-CoV-2-Mpro")
    scores = scorer.score([data], use_dask=use_dask, return_df=return_df)
    assert len(scores) == 1
