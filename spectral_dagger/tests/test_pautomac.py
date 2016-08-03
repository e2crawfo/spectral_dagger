import pytest
import numpy as np

from spectral_dagger.datasets.pautomac import (
    pautomac_available, problem_indices,
    load_pautomac_model, pautomac_score,
    make_pautomac_like)


@pytest.mark.skipif(
    not pautomac_available(), reason="PAutomaC problems were not found.")
def test_load_pautomac():
    for idx in problem_indices():
        load_pautomac_model(idx)


@pytest.mark.skipif(
    not pautomac_available(), reason="PAutomaC problems were not found.")
def test_pautomac_score():
    """ Test against the references given in Balle, Hamilton, Pineau 2014. """

    targets = [
        (1, 29.90),
        (14, 116.80),
        (33, 31.87),
        (45, 24.04),
        (29, 24.03),
        (39, 10.00),
        (43, 32.64),
        (46, 11.98),
        (6, 66.96),
        (7, 51.22),
        (27, 42.43),
        (42, 16.00)]

    for problem_idx, target in targets:
        model = load_pautomac_model(problem_idx)
        score = pautomac_score(model, problem_idx)

        assert np.isclose(score, target, atol=0.05, rtol=0.0)


@pytest.mark.parametrize("kind", ["pfa", "hmm"])
@pytest.mark.parametrize("halts", [False, True, 0.9, 2])
def test_make_pautomac_like(kind, halts):
    n_states = 10
    n_symbols = 5
    symbol_density = 0.5
    transition_density = 0.5

    model = make_pautomac_like(
        kind, n_states, n_symbols,
        symbol_density, transition_density,
        alpha=1.0, halts=halts)
    model.sample_episodes(5, horizon=np.inf if halts else 5)
