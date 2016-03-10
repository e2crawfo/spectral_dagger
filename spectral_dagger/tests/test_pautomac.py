import pytest
import numpy as np

from spectral_dagger.datasets.pautomac import (
    pautomac_available, problem_indices,
    load_pautomac_model, pautomac_score)


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
