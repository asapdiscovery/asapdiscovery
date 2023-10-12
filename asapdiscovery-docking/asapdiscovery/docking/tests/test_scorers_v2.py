import os

import pytest

from asapdiscovery.docking.docking_v2 import POSITDocker
from asapdiscovery.docking.scorer_v2 import (
    ChemGauss4Scorer,
    GATScorer,
    SchnetScorer,
    MetaScorer,
)


@pytest.fixture(scope="session")
def results(docking_input_pair_simple):
    docker = POSITDocker()
    results = docker.dock([docking_input_pair_simple])
    return results


@pytest.fixture(scope="session")
def results_multi(results):
    return [results[0], results[0]]


@pytest.mark.parametrize("use_dask", [True, False])
def test_chemgauss_scorer(results_multi, use_dask):
    scorer = ChemGauss4Scorer()
    scores = scorer.score(results_multi, use_dask=use_dask)
    assert len(scores) == 2
    assert scores[0].score_type == "chemgauss4"
    assert scores[0].score < 0.0


def test_chemgauss_scorer_df(results_multi):
    scorer = ChemGauss4Scorer()
    scores = scorer.score(results_multi, return_df=True)
    print(scores)


@pytest.mark.parametrize("use_dask", [True, False])
def test_gat_scorer(results_multi, use_dask):
    scorer = GATScorer.from_latest_by_target("SARS-CoV-2-Mpro")
    scores = scorer.score(results_multi, use_dask=use_dask)
    assert len(scores) == 2
    assert scores[0].score_type == "GAT"
    assert scores[0].score > 0.0


@pytest.mark.parametrize("use_dask", [True, False])
def test_schnet_scorer(results_multi, use_dask):
    scorer = SchnetScorer.from_latest_by_target("SARS-CoV-2-Mpro")
    scores = scorer.score(results_multi, use_dask=use_dask)
    assert len(scores) == 2
    assert scores[0].score_type == "schnet"
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
    assert len(scores) == 1
    assert len(scores[0]) == 3
    assert scores[0][0].score_type == "chemgauss4"
    assert scores[0][1].score_type == "GAT"
    assert scores[0][2].score_type == "schnet"
