from spectral_dagger.spectral import SpectralPSR
from spectral_dagger.spectral.hankel import single_obs_basis
from spectral_dagger.spectral.hankel import true_hankel_for_hmm
from spectral_dagger.hmm import HMM
from spectral_dagger.utils.math import normalize

import pytest


@pytest.fixture
def simple_hmm():
    n_obs = 2
    n_states = 2

    observations = range(n_obs)
    states = range(n_states)

    O = normalize([[1, 1], [.1, 1]], ord=1)
    T = normalize([[8, 2], [2, 8]], ord=1)

    init_dist = normalize([1, 1], ord=1)

    return HMM(observations, states, T, O, init_dist)


@pytest.mark.parametrize(
    'estimator', ['string', 'prefix', 'substring', 'single'])
def test_spectral_hmm(
        simple_hmm, estimator, n_samples=100, horizon=3, m=None):

    print "*" * 80
    samples = [simple_hmm.sample_trajectory(horizon) for i in range(n_samples)]
    psr = SpectralPSR(simple_hmm.observations)

    if m is None:
        m = hmm.n_states

    basis = None
    if estimator == 'single':
        basis = single_obs_basis(simple_hmm.observations, True)
        estimator = 'substring'

    psr.fit(samples, m, estimator, basis=basis)

    for o in simple_hmm.observations:
        print psr.B_o[o]

    print(psr.get_obs_prob(0))
    print(psr.get_obs_prob(1))

    print psr.get_prediction()
    print psr.get_seq_prob([1, 1])
    print psr.get_seq_prob([1, 0])
    print psr.get_seq_prob([0, 1])
    print psr.get_seq_prob([0, 0])

    basis = (psr.prefix_dict, psr.suffix_dict)

    print "True"
    print true_hankel_for_hmm(hmm, basis, length=horizon, estimator=estimator)

    print "Estimated"
    print psr.hankel


if __name__ == "__main__":
    hmm = simple_hmm()
    dimension = 6

    test_spectral_hmm(hmm, 'string', m=dimension)
    test_spectral_hmm(hmm, 'prefix', m=dimension)
    test_spectral_hmm(hmm, 'substring', m=dimension)
    test_spectral_hmm(hmm, 'single', m=dimension)

    print "HMM" + "*" * 80
    print(hmm.get_seq_prob([0]))
    print(hmm.get_seq_prob([1]))

    print hmm.get_seq_prob([1, 1])
    print hmm.get_seq_prob([1, 0])
    print hmm.get_seq_prob([0, 1])
    print hmm.get_seq_prob([0, 0])

    print hmm.get_subsequence_expectation([0, 1], 4)
