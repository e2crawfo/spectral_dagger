import pytest

from spectral_dagger.hmm.pautomac import pautomac_available, problem_indices
from spectral_dagger.hmm.pautomac import load_pautomac_model


@pytest.mark.skipif(
    not pautomac_available(), reason="Pautomac problems were not found.")
def test_load_pautomac():
    for idx in problem_indices():
        load_pautomac_model(idx)
