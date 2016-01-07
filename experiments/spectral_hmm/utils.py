import numpy as np
from sklearn.utils.extmath import randomized_svd

from spectral_dagger.spectral.hankel import fixed_length_basis, top_k_basis
from spectral_dagger.utils.math import normalize
from spectral_dagger.hmm import HMM, dummy_hmm, bernoulli_hmm
from spectral_dagger.spectral import SpectralPSR
from spectral_dagger.spectral import hankel
from spectral_dagger.envs import EgoGridWorld


def perplexity_score(psr, testing_data):
    return psr.get_perplexity(testing_data, base=2)


def make_prediction_score(hmm, horizon=2, n_samples=10000, ord=2):
    probs = [
        hmm.get_delayed_seq_prob([o], horizon-1) for o in hmm.observations]

    assert np.isclose(sum(probs), 1.0)

    def prediction_score(psr, testing_data=None):
        psr_probs = np.array([
            psr.get_delayed_seq_prob([o], horizon-1)
            for o in hmm.observations])

        return np.linalg.norm(probs - psr_probs, ord=ord)

    return prediction_score


def choose_basis(hmm, training_data, estimator, basis_max_length):
    if basis_max_length < 0:
        basis = fixed_length_basis(hmm.observations, -basis_max_length)
    else:
        basis = top_k_basis(
            training_data, np.inf, estimator, max_length=basis_max_length)

    return basis


def fit_estimated_svd(
        hmm, training_data, dimension, estimator, basis_max_length):

    basis = choose_basis(hmm, training_data, estimator, basis_max_length)

    psr = SpectralPSR(hmm.observations)
    psr.fit(training_data, dimension, estimator, basis=basis)

    return psr


def fit_true_svd(
        hmm, training_data, dimension, estimator, basis_max_length):

    basis = choose_basis(hmm, training_data, estimator, basis_max_length)

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
    basis = choose_basis(hmm, training_data, estimator, basis_max_length)

    tuple_basis = (
        tuple(basis[0].keys()), tuple(basis[1].keys()))

    horizon = len(training_data[0])
    key = (estimator, tuple_basis, horizon)
    hankels = all_hankels.get(key)

    if hankels is None:
        hankels = hankel.true_hankel_for_hmm(
            hmm, basis, horizon, estimator, full=True)
        all_hankels[key] = hankels

    psr = SpectralPSR(hmm.observations)
    psr.fit(training_data, dimension, estimator, basis=basis, hankels=hankels)

    return psr


def scan_dimensions(
        training_data, testing_data, hmm, dimension_seq,
        estimator, basis_max_length, score_func, fit_func):

    results = []

    for dim in dimension_seq:
        psr = fit_func(hmm, training_data, dim, estimator, basis_max_length)
        score = score_func(psr, testing_data)

        results.append((psr, dim, score))

    return min(results, key=lambda x: x[2])


def grid_world_hmm(world=None, n_colors=1, noise=0.2):
    if world is None:
        world = np.array([
            ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x'],
            ['x', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x', ' ', ' ', ' ', 'x'],
            ['x', ' ', 'G', 'x', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x'],
            ['x', ' ', ' ', 'x', ' ', ' ', ' ', ' ', ' ', ' ', 'x', 'x', 'x', 'x'],
            ['x', ' ', ' ', 'x', 'S', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x'],
            ['x', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x', ' ', ' ', ' ', 'x'],
            ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x']]
        )

    env = EgoGridWorld(n_colors, world, noise=noise)

    # open loop policy
    action_dist = normalize(np.ones(env.n_actions), ord=1)

    T = np.tensordot(env.T, action_dist, ((0,), (0,)))
    O = env.O[0, :, :]
    O.squeeze()

    init_dist = env.init_dist.copy()

    n_states = T.shape[0]
    n_obs = O.shape[1]

    observations = range(n_obs)
    states = range(n_states)

    hmm = HMM(observations, states, T, O, init_dist)

    return hmm


if __name__ == "__main__":
    env = 1

    data_seed = np.random.randint(10000)

    if env == 0:
        hmm = grid_world_hmm()

        horizon = 3
        n_samples = 100
        dimension_seq = range(20, hmm.n_states+1)
        # dimension_seq = range(20, 40)
    elif env == 1:
        model_seed = np.random.randint(10000)
        model_rng = np.random.RandomState(model_seed)

        n_states = 1
        n_obs = 5

        hmm = bernoulli_hmm(n_states, n_obs, model_rng)

        horizon = 3
        n_samples = 1000
        dimension_seq = range(1, 2 * n_states+1)
    else:
        n_states = 2
        hmm = dummy_hmm(n_states)

        n_samples = 100
        horizon = 3
        dimension_seq = range(1, 2 * n_states+1)

    basis_max_length = horizon
    #  basis_max_length = np.floor((horizon - 1) / 2.0)

    print(hmm.sample_trajectory(10))

    training_samples = [
        hmm.sample_trajectory(horizon) for i in range(n_samples)]

    testing_samples = [hmm.sample_trajectory(horizon) for i in range(10)]

    # score_func = perplexity_score
    score_func = make_prediction_score(hmm)

    # Do spectral learning with true hankels
    for e in ['string', 'prefix', 'substring']:
        model, n, score = scan_dimensions(
            training_samples, testing_samples, hmm, dimension_seq,
            e, basis_max_length, score_func, fit_true_hankels)

        print "v" * 40
        print "True Hankels with estimator %s" % e
        print "Dimension: ", n
        print "Score: ", score
        print "v" * 40

    # Do vanilla spectral learning
    for e in ['string', 'prefix', 'substring']:
        model, n, score = scan_dimensions(
            training_samples, testing_samples, hmm, dimension_seq,
            e, basis_max_length, score_func, fit_estimated_svd)

        print "^" * 40
        print "Estimated SVD with estimator %s" % e
        print "Dimension: ", n
        print "Score: ", score
        print "^" * 40

    # Do spectral learning with true U, S, V
    for e in ['string', 'prefix', 'substring']:
        model, n, score = scan_dimensions(
            training_samples, testing_samples, hmm, dimension_seq,
            e, basis_max_length, score_func, fit_true_svd)

        print "@" * 40
        print "True SVD with estimator %s" % e
        print "Dimension: ", n
        print "Score: ", score
        print "@" * 40