import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd

from spectral_dagger.spectral.hankel import single_obs_basis, top_k_basis
from spectral_dagger.utils.math import normalize
from spectral_dagger.hmm import HMM
from spectral_dagger.spectral import SpectralPSR
from spectral_dagger.spectral import hankel
from spectral_dagger.envs import EgoGridWorld

if 0:
    world = np.array([
        ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x'],
        ['x', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x', ' ', ' ', ' ', 'x'],
        ['x', ' ', 'G', 'x', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x'],
        ['x', ' ', ' ', 'x', ' ', ' ', ' ', ' ', ' ', ' ', 'x', 'x', 'x', 'x'],
        ['x', ' ', ' ', 'x', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x'],
        #['x', ' ', ' ', 'x', 'S', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x'],
        ['x', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x', ' ', ' ', ' ', 'x'],
        ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x']]
    )

    n_colors = 1

    env = EgoGridWorld(n_colors, world, noise=0.2)

    # open loop policy
    action_dist = normalize(np.ones(env.n_actions), ord=1)

    T = np.tensordot(env.T, action_dist, ((0,), (0,)))
    O = env.O[0, :, :]
    O.squeeze()

    init_dist = env.init_dist.copy()

    n_states = T.shape[0]
    n_obs = O.shape[1]

    print n_states
    print n_obs

    observations = range(n_obs)
    states = range(n_states)

    hmm = HMM(observations, states, T, O, init_dist)

    data_seed = np.random.randint(10000)
    np.random.seed(data_seed)
    print(hmm.sample_trajectory(10))

    horizon = 3
    n_samples = 100
    basis_max_length = np.floor((horizon - 1) / 2.0)
    dimension_seq = [30, hmm.n_states]
    # dimension_seq = range(20, 40)
elif 0:
    model_seed = np.random.randint(10000)
    data_seed = np.random.randint(10000)
    np.random.seed(model_seed)

    n_states = 1
    n_obs = 5

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
    n_samples = 1000
    basis_max_length = np.floor((horizon - 1) / 2.0)
    dimension_seq = range(1, 2 * n_states+1)
else:
    model_seed = np.random.randint(10000)
    data_seed = np.random.randint(10000)
    np.random.seed(model_seed)

    n_states = n_obs = 5

    observations = range(n_obs)
    states = range(n_states)

    O = np.eye(n_states)

    # T = np.random.binomial(1, 0.5, (n_states, n_states))
    # for row in T:
    #     if sum(row) == 0:
    #         row[:] = 1.0

    # T = normalize(T, ord=1, conservative=True)

    T = np.eye(n_states)
    init_dist = normalize(np.ones(n_states), ord=1, conservative=True)

    hmm = HMM(observations, states, T, O, init_dist)

    np.random.seed(data_seed)
    print(hmm.sample_trajectory(10))

    n_samples = 10000
    horizon = 10
    basis_max_length = np.floor((horizon - 1) / 2.0)
    dimension_seq = range(1, 2 * n_states+1)


def perplexity_score(psr, testing_data):
    return psr.get_perplexity(testing_data, base=2)


def make_prediction_score(hmm, horizon=2, n_samples=10000, ord=2):
    # Construct vector of probabilities at timestep t
    samples = [hmm.sample_trajectory(horizon) for i in range(n_samples)]
    probs = np.zeros(hmm.n_observations)

    for i, obs in enumerate(hmm.observations):
        for sample in samples:
            if sample[-1] == obs:
                probs[i] += 1.0 / n_samples

    assert np.isclose(sum(probs), 1.0)
    print(probs)

    def prediction_score(psr, testing_data=None):
        psr_probs = np.array([
            psr.get_delayed_seq_prob([o], horizon-1)
            for o in hmm.observations])

        return np.linalg.norm(probs - psr_probs, ord=ord)

    return prediction_score


def fit_estimated_svd(
        hmm, training_data, dimension, estimator, basis_max_length):

    if estimator == 'single':
        basis = single_obs_basis(hmm.observations, True)
        estimator = 'prefix'
    else:
        basis = top_k_basis(training_data, np.inf, estimator, basis_max_length)

    psr = SpectralPSR(hmm.observations)
    psr.fit(training_data, dimension, estimator, basis=basis)

    return psr


def fit_true_svd(
        hmm, training_data, dimension, estimator, basis_max_length):

    if estimator == 'single':
        basis = single_obs_basis(hmm.observations, True)
        estimator = 'prefix'
    else:
        basis = top_k_basis(training_data, np.inf, estimator, basis_max_length)

    psr = SpectralPSR(hmm.observations)

    true_hankel = hankel.true_hankel_for_hmm(hmm, basis, horizon, estimator)

    n_oversamples = 10
    n_iter = 5

    svd = randomized_svd(true_hankel, dimension, n_oversamples, n_iter)
    # svd = np.linalg.svd(true_hankel.toarray())

    psr.fit(training_data, dimension, estimator, basis=basis, svd=svd)

    return psr

all_hankels = {}


def fit_true_hankels(
        hmm, training_data, dimension, estimator, basis_max_length):

    if estimator == 'single':
        basis = single_obs_basis(hmm.observations, True)
        estimator = 'prefix'
    else:
        basis = top_k_basis(training_data, np.inf, estimator, basis_max_length)

    hankels = all_hankels.get(estimator)

    if hankels is None:
        hankels = hankel.true_hankel_for_hmm(
            hmm, basis, len(training_data[0]), estimator, full=True)
        all_hankels[estimator] = hankels

    psr = SpectralPSR(hmm.observations)
    psr.fit(training_data, dimension, estimator, basis=basis, hankels=hankels)

    return psr


def scan_n_states(
        training_data, testing_data, hmm, dimension_seq,
        estimator, basis_max_length, score_func, fit_func):

    results = []

    for dim in dimension_seq:
        psr = fit_func(hmm, training_data, dim, estimator, basis_max_length)
        score = score_func(psr, testing_data)

        results.append((psr, dim, score))

    return min(results, key=lambda x: x[2])


training_samples = [hmm.sample_trajectory(horizon) for i in range(n_samples)]
testing_samples = [hmm.sample_trajectory(horizon) for i in range(10)]

# score_func = perplexity_score
score_func = make_prediction_score(hmm)


# Do spectral learning with true hankels
for e in ['string', 'prefix', 'substring', 'single']:
    model, n, score = scan_n_states(
        training_samples, testing_samples, hmm, dimension_seq,
        e, basis_max_length, score_func, fit_true_hankels)

    print "v" * 40
    print "True Hankels with estimator %s" % e
    print "Dimension: ", n
    print "Score: ", score
    print "v" * 40


# Do vanilla spectral learning
for e in ['string', 'prefix', 'substring', 'single']:
    model, n, score = scan_n_states(
        training_samples, testing_samples, hmm, dimension_seq,
        e, basis_max_length, score_func, fit_estimated_svd)

    print "^" * 40
    print "Estimated SVD with estimator %s" % e
    print "Dimension: ", n
    print "Score: ", score
    print "^" * 40


# Do spectral learning with true U, S, V
for e in ['string', 'prefix', 'substring', 'single']:
    model, n, score = scan_n_states(
        training_samples, testing_samples, hmm, dimension_seq,
        e, basis_max_length, score_func, fit_true_svd)

    print "@" * 40
    print "True SVD with estimator %s" % e
    print "Dimension: ", n
    print "Score: ", score
    print "@" * 40