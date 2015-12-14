import numpy as np

from spectral_dagger.spectral import SpectralPSR
from spectral_dagger.hmm import HMM
from spectral_dagger.utils.math import normalize


def test_spectral_hmm():
    n_obs = 2
    n_states = 2

    observations = range(n_obs)
    states = range(n_states)

    O = normalize([[1, 1], [.1, 1]], ord=1)
    T = normalize([[8, 2], [2, 8]], ord=1)

    init_dist = normalize([1, 1], ord=1)

    hmm = HMM(observations, states, T, O, init_dist)

    n_samples = 1000
    horizon = 2

    samples = [hmm.sample_trajectory(horizon) for i in range(n_samples)]
    print(samples)
    psr = SpectralPSR(observations)

    psr.fit(samples, n_states, 1000)

    print(psr.get_obs_prob(0))
    print(psr.get_obs_prob(1))

    print psr.get_prediction()
    print psr.get_seq_prob([1, 1])
    print psr.get_seq_prob([1, 0])
    print psr.get_seq_prob([0, 1])
    print psr.get_seq_prob([0, 0])


if __name__ == "__main__":
    test_spectral_hmm()
