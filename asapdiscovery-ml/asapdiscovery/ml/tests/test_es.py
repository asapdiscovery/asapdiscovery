import numpy as np
import pytest
from asapdiscovery.ml.es import (
    BestEarlyStopping,
    ConvergedEarlyStopping,
    PatientConvergedEarlyStopping,
)


@pytest.fixture()
def best_losses():
    return [0.1] + [1] * 100


@pytest.fixture()
def converged_losses():
    varying_losses = (np.sin(-np.linspace(0, 2 * np.pi, 20)) + 1) * 100
    converged_losses = np.random.default_rng().uniform(0.95, 1.05, size=100)

    return np.concatenate([varying_losses, converged_losses])


def test_best_es_no_burnin(best_losses):
    es = BestEarlyStopping(10)

    for i, loss in enumerate(best_losses):
        if es.check(i, loss, {"epoch": i}):
            break

    assert i == 10
    assert es.best_loss == 0.1
    assert es.best_wts == {"epoch": 0}


def test_best_es_burnin(best_losses):
    es = BestEarlyStopping(10, burnin=20)

    for i, loss in enumerate(best_losses):
        if es.check(i, loss, {"epoch": i}):
            break

    assert i == 20
    assert es.best_loss == 0.1
    assert es.best_wts == {"epoch": 0}


def test_converged_es_no_burnin(converged_losses):
    es = ConvergedEarlyStopping(20, 0.1)

    for i, loss in enumerate(converged_losses):
        if es.check(i, loss):
            break

    assert i == 39


def test_converged_es_burnin(converged_losses):
    es = ConvergedEarlyStopping(20, 0.1, 50)

    for i, loss in enumerate(converged_losses):
        if es.check(i, loss):
            break

    assert i == 50


def test_patient_converged_es_no_burnin(converged_losses):
    es = PatientConvergedEarlyStopping(20, 0.1, 20)

    for i, loss in enumerate(converged_losses):
        if es.check(i, loss, {"epoch": i}):
            break

    assert i == 59
    assert es.converged_epoch == 39
    assert es.converged_wts == {"epoch": 39}


def test_patient_converged_es_burnin(converged_losses):
    es = PatientConvergedEarlyStopping(20, 0.1, 20, 70)

    for i, loss in enumerate(converged_losses):
        if es.check(i, loss, {"epoch": i}):
            break

    assert i == 70
    assert es.converged_epoch == 39
    assert es.converged_wts == {"epoch": 39}
