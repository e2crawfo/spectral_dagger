import numpy as np

from spectral_dagger.spectral import SpectralPSR
from spectral_dagger.hmm import HMM
from spectral_dagger.utils.math import normalize


def test_spectral_hmm():
    n_obs = 2
    n_states = 2

    observations = range(n_obs)
    states = range(n_states)

    O = normalize([[8, 2], [2, 8]], ord=1)
    T = normalize([[8, 2], [2, 8]], ord=1)

    init_dist = normalize([0, 1], ord=1)

    hmm = HMM(observations, states, T, O, init_dist)

    n_samples = 1000
    horizon = 3

    samples = [hmm.sample_trajectory(horizon) for i in range(n_samples)]
    psr = SpectralPSR(observations)

    psr.fit(samples, 100, n_states)

    print(psr.get_obs_prob(0))
    print(psr.get_obs_prob(1))

    print psr.get_prediction()


if __name__ == "__main__":
    test_spectral_hmm()
