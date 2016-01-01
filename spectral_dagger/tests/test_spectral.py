from spectral_dagger.spectral import SpectralPSR
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


def test_spectral_hmm_string(simple_hmm):
    n_samples = 1000
    horizon = 3

    samples = [simple_hmm.sample_trajectory(horizon) for i in range(n_samples)]
    psr = SpectralPSR(simple_hmm.observations)

    psr.fit(samples, simple_hmm.n_states, 'string')

    for o in simple_hmm.observations:
        print psr.B_o[o]

    print(psr.get_obs_prob(0))
    print(psr.get_obs_prob(1))

    print psr.get_prediction()
    print psr.get_seq_prob([1, 1])
    print psr.get_seq_prob([1, 0])
    print psr.get_seq_prob([0, 1])
    print psr.get_seq_prob([0, 0])


def test_spectral_hmm_prefix(simple_hmm):
    n_samples = 10000
    horizon = 2

    samples = [simple_hmm.sample_trajectory(horizon) for i in range(n_samples)]
    psr = SpectralPSR(simple_hmm.observations)

    psr.fit(samples, simple_hmm.n_states, 'prefix')

    print(psr.get_obs_prob(0))
    print(psr.get_obs_prob(1))

    print psr.get_prediction()
    print psr.get_seq_prob([1, 1])
    print psr.get_seq_prob([1, 0])
    print psr.get_seq_prob([0, 1])
    print psr.get_seq_prob([0, 0])


def test_spectral_hmm_substring(simple_hmm):
    n_samples = 10000
    horizon = 2

    samples = [simple_hmm.sample_trajectory(horizon) for i in range(n_samples)]
    psr = SpectralPSR(simple_hmm.observations)

    psr.fit(samples, simple_hmm.n_states, 'substring')

    print(psr.get_obs_prob(0))
    print(psr.get_obs_prob(1))

    print psr.get_prediction()
    print psr.get_seq_prob([1, 1])
    print psr.get_seq_prob([1, 0])
    print psr.get_seq_prob([0, 1])
    print psr.get_seq_prob([0, 0])

if __name__ == "__main__":
    hmm = simple_hmm()
    test_spectral_hmm_string(hmm)
    #test_spectral_hmm_prefix(hmm)
    #test_spectral_hmm_substring(hmm)
