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
    print(type(data))
    scorer = ChemGauss4Scorer()
    scores = scorer.score([data], use_dask=use_dask, return_df=return_df)
    assert len(scores) == 1


@pytest.mark.parametrize("use_dask", [True, False])
def test_chemgauss_scorer_complex(complex_simple, use_dask):
    scorer = ChemGauss4Scorer()
    scores = scorer.score([complex_simple], return_df=True, use_dask=use_dask)
    assert len(scores) == 1


@pytest.mark.parametrize("use_dask", [True, False])
def test_chemgauss_scorer_path(pdb_simple, use_dask):
    scorer = ChemGauss4Scorer()
    scores = scorer.score([pdb_simple], return_df=True, use_dask=use_dask)
    assert len(scores) == 1


@pytest.mark.parametrize("use_dask", [True, False])
def test_gat_scorer(results_multi, use_dask):
    scorer = GATScorer.from_latest_by_target("SARS-CoV-2-Mpro")
    scores = scorer.score(results_multi, use_dask=use_dask)
    assert len(scores) == 2
    assert scores[0].score_type == "GAT"
    assert scores[0].score > 0.0


@pytest.mark.parametrize("use_dask", [True, False])
def test_gat_scorer_smiles(use_dask):
    scorer = GATScorer.from_latest_by_target("SARS-CoV-2-Mpro")
    scores = scorer.score(["CCCC"], use_dask=use_dask)
    assert scores[0].score > 0.0


@pytest.mark.parametrize("use_dask", [True, False])
def test_gat_scorer_ligand(ligand, use_dask):
    scorer = GATScorer.from_latest_by_target("SARS-CoV-2-Mpro")
    scores = scorer.score([ligand], use_dask=use_dask)
    assert scores[0].score > 0.0


@pytest.mark.parametrize("use_dask", [True, False])
def test_schnet_scorer(
    results_multi,
    use_dask,
):
    scorer = SchnetScorer.from_latest_by_target("SARS-CoV-2-Mpro")
    scores = scorer.score(results_multi, use_dask=use_dask)
    assert scores[0].score_type == "schnet"


@pytest.mark.parametrize("use_dask", [True, False])
def test_schnet_complex(complex_simple, use_dask):
    scorer = SchnetScorer.from_latest_by_target("SARS-CoV-2-Mpro")
    scores = scorer.score([complex_simple], use_dask=use_dask)
    assert len(scores) == 1
    assert scores[0].score_type == "schnet"


@pytest.mark.parametrize("use_dask", [True, False])
def test_schnet_path(pdb_simple, use_dask):
    scorer = SchnetScorer.from_latest_by_target("SARS-CoV-2-Mpro")
    scores = scorer.score([pdb_simple], use_dask=use_dask)
    assert len(scores) == 1
    assert scores[0].score_type == "schnet"


@pytest.mark.parametrize("use_dask", [True, False])
def test_e3nn_scorer(results_multi, use_dask):
    scorer = E3NNScorer.from_latest_by_target("SARS-CoV-2-Mpro")
    scores = scorer.score(results_multi, use_dask=use_dask)
    assert len(scores) == 2
    assert scores[0].score_type == "e3nn"
    assert scores[0].score > 0.0


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
        ]
    )

    scores = scorer.score(results_multi, return_df=True)
    assert len(scores) == 2  # 3 scorers for each of 2 inputs


@pytest.mark.parametrize("use_dask", [True, False])
def test_FINT_scorer(results_multi, use_dask):
    scorer = FINTScorer(target="SARS-CoV-2-Mpro")
    scores = scorer.score(results_multi, use_dask=use_dask)
    assert len(scores) == 2
    assert scores[0].score_type == "FINT"
    assert scores[0].score > 0.0
    assert scores[0].score <= 1.0
