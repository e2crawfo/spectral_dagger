import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from spectral_dagger.spectral.hankel import single_obs_basis
from spectral_dagger.utils.math import normalize
from spectral_dagger.hmm import HMM
from spectral_dagger.spectral import SpectralPSR


def test_hmm():

    model_seed = 1
    data_seed = np.random.randint(10000)
    np.random.seed(model_seed)

    n_states = 5
    n_obs = 6

    observations = range(n_obs)
    states = range(n_states)

    O = np.random.binomial(1, 0.5, (n_states, n_obs))
    for row in O:
        if sum(row) == 0:
            row[:] = 1.0

    O = normalize(O, ord=1, conservative=True)

    T = np.random.binomial(1, 0.5, (n_states, n_states))
    for row in T:
        if sum(row) == 0:
            row[:] = 1.0

    T = normalize(T, ord=1, conservative=True)

    init_dist = normalize(np.ones(n_states), ord=1, conservative=True)

    hmm = HMM(observations, states, T, O, init_dist)

    np.random.seed(data_seed)
    print(hmm.sample_trajectory(10))

    horizon = 3
    test_samples = [hmm.sample_trajectory(horizon) for i in range(1000)]
    results = []

    estimators = ['string', 'prefix', 'substring', 'single']
    sizes = np.linspace(1000, 10000, 5).astype('i')
    dimensions = np.linspace(n_states, 2 * n_states, 5).astype('i')

    for estimator in estimators:
        for m in dimensions:
            for n_samples in sizes:

                print "*" * 80
                samples = [hmm.sample_trajectory(horizon) for i in range(n_samples)]
                psr = SpectralPSR(hmm.observations)

                if m is None:
                    m = hmm.n_states

                basis = None
                if estimator == 'single':
                    basis = single_obs_basis(hmm.observations, True)
                    estimator = 'substring'

                psr.fit(samples, m, estimator, basis=basis)

                llh = psr.get_log_likelihood(test_samples, base=2)
                print "$" * 80
                print "n_samples: ", n_samples
                perplexity = 2**(-llh)
                print "Perplexity: ", perplexity

                results.append(dict(
                    n_samples=n_samples, estimator=estimator,
                    dimension=m, perplexity=perplexity))

    results = pd.DataFrame.from_records(results)

    for estimator in estimators:
        for m in dimensions:
            data = results[
                (results['estimator'] == estimator)
                & (results['dimension'] == m)]

            plt.plot(data['n_samples'], data['perplexity'], label='%s, Dim=%d' % (estimator, m))

    plt.ylim((0.0, 100))
    plt.legend(loc=2)
    plt.show()

if __name__ == "__main__":
    test_hmm()