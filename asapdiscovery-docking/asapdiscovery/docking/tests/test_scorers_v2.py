import os

import pytest

from asapdiscovery.docking.docking_v2 import POSITDocker
from asapdiscovery.docking.scorer_v2 import ChemGauss4Scorer, GATScorer, SchnetScorer


@pytest.fixture(scope="session")
def results(docking_input_pair_simple):
    docker = POSITDocker()
    results = docker.dock([docking_input_pair_simple])
    return results


def test_chemgauss_scorer(results):
    scorer = ChemGauss4Scorer()
    scores = scorer.score(results)
    assert len(scores) == 1
    assert scores[0].score_type == "chemgauss4"
    assert scores[0].score < 0.0


def test_gat_scorer(results):
    scorer = GATScorer.from_latest_by_target("SARS-CoV-2-Mpro")
    scores = scorer.score(results)
    assert len(scores) == 1
    assert scores[0].score_type == "GAT"
    assert scores[0].score > 0.0


def test_schnet_scorer(results):
    scorer = SchnetScorer.from_latest_by_target("SARS-CoV-2-Mpro")
    scores = scorer.score(results)
    assert len(scores) == 1
    assert scores[0].score_type == "Schnet"
    assert scores[0].score > 0.0
