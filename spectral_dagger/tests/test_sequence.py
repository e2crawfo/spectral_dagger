import pytest
import numpy as np
from scipy import stats
from scipy.linalg import orth
from itertools import product
from collections import defaultdict
import six
from sklearn.utils import check_random_state

from spectral_dagger.sequence import SpectralSA, CompressedSA
from spectral_dagger.sequence import SpectralKernelSA, KernelInfo
from spectral_dagger.sequence import top_k_basis, fixed_length_basis
from spectral_dagger.sequence import ContinuousHMM
from spectral_dagger.sequence import ExpMaxSA
from spectral_dagger.sequence import ConvexOptSA
from spectral_dagger.sequence import MixtureSeqGen
from spectral_dagger.sequence import HMM, MarkovChain, AdjustedMarkovChain
from spectral_dagger.sequence import GMMHMM
from spectral_dagger.utils.math import normalize, rmse
from spectral_dagger.datasets.pautomac import make_pautomac_like


@pytest.fixture
def simple_hmm():
    O = normalize([[1, 1], [.1, 1]], ord=1)
    T = normalize([[8, 2], [2, 8]], ord=1)

    init_dist = normalize([1, 1], ord=1)
    return HMM(init_dist, T, O)


@pytest.fixture
def reduced_rank_hmm(random_state=None):
    rng = check_random_state(random_state)

    n = 6
    r = 3

    T_cols = rng.binomial(100, 0.5, (n, r))
    T_cols[T_cols == 0] = 1

    T = np.zeros((n, n))
    for i in range(n):
        T[:, i] = T_cols[:, rng.randint(r)]

    T = normalize(T, ord=1, conservative=True)

    rank = np.linalg.matrix_rank(T)
    assert rank <= r

    O = np.abs(orth(rng.randn(n, n)))
    O = normalize(O, ord=1, conservative=True)

    return HMM(normalize(np.ones(n), ord=1), T, O)


def do_test_hmm_learning(
        hmm, learning_alg, horizon=3, n_samples=1000, tol=0.1, **kwargs):

    samples = hmm.sample_episodes(n_samples, horizon=horizon)

    sa = learning_alg(hmm, samples, **kwargs)

    test_seqs = [[0], [1], [0, 0], [0, 1], [1, 0], [1, 1]]

    error = 0

    print(("*" * 20))
    for seq in test_seqs:
        ground_truth = hmm.string_prob(seq)
        pred = sa.string_prob(seq)
        print(("Seq: ", seq))
        print("String estimate:")
        print(("Ground truth: %f" % ground_truth))
        print(("Prediction: %f" % pred))
        if n_samples >= 4000:
            assert np.isclose(ground_truth, pred, atol=tol, rtol=0.0)

        error += np.abs(ground_truth - pred)

        ground_truth = hmm.prefix_prob(seq)
        pred = sa.prefix_prob(seq)
        print(("Seq: ", seq))
        print("Prefix estimate:")
        print(("Ground truth: %f" % ground_truth))
        print(("Prediction: %f" % pred))
        if n_samples >= 4000:
            assert np.isclose(ground_truth, pred, atol=tol, rtol=0.0)

        error += np.abs(ground_truth - pred)

    return error


@pytest.mark.parametrize("learning_alg", [SpectralSA, CompressedSA])
def test_spectral_like(learning_alg):
    def learn(hmm, samples, dimension, estimator, basis=None):

        if learning_alg == CompressedSA:
            estimator = 'prefix'

        if basis is None:
            basis = top_k_basis(samples, np.inf, estimator)

        sa = learning_alg(dimension, hmm.n_observations, estimator=estimator)
        sa.fit(samples, basis=basis)
        return sa

    seed = 10
    np.random.seed(seed)

    print(("Learning with: %s" % learning_alg))
    hmm = simple_hmm()
    dimension = 6

    print("String estimator.")
    do_test_hmm_learning(
        hmm, learn, estimator='string', dimension=dimension)
    print("Prefix estimator.")
    do_test_hmm_learning(
        hmm, learn, estimator='prefix', dimension=dimension)
    print("Substring estimator.")
    do_test_hmm_learning(
        hmm, learn, estimator='substring', dimension=dimension)

    rr_hmm = reduced_rank_hmm()
    dimension = np.linalg.matrix_rank(rr_hmm.T)

    params = product([1, 2], [3, 5], [True, False])

    for bl, horizon, include_empty in params:
        if 2 * bl + 1 <= horizon:
            print(("Basis with length %d, horizon=%d, "
                  "include_empty: %r." % (bl, horizon, include_empty)))
            basis = fixed_length_basis(rr_hmm.observations, bl, include_empty)
            do_test_hmm_learning(
                rr_hmm, learn, horizon=horizon,
                estimator='prefix', dimension=dimension, basis=basis)


@pytest.mark.skipif(True, reason="Incomplete implementation.")
def test_em():
    def learn(hmm, samples):
        validate_samples = hmm.sample_episodes(
            len(samples), horizon=len(samples[0]))
        em_sa = ExpMaxSA(hmm.n_states, hmm.n_observations)
        em_sa.fit(samples, validate_samples)
        return em_sa

    seed = 10
    np.random.seed(seed)

    hmm = simple_hmm()
    do_test_hmm_learning(hmm, learn)

    rr_hmm = reduced_rank_hmm()
    do_test_hmm_learning(rr_hmm, learn, horizon=5)


@pytest.mark.skipif(True, reason="Incomplete implementation.")
def test_convex_opt():
    horizon = 3
    rank_tol = 1e-7
    tau = 0.001

    def learn(hmm, samples, dimension, basis=None):
        if basis is None:
            basis = top_k_basis(samples, np.inf, 'prefix')
        sa = ConvexOptSA(hmm.observations, estimator='prefix')
        sa.fit(samples, dimension, basis=basis)
        return sa

    seed = 10
    np.random.seed(seed)

    hmm = simple_hmm()

    basis = fixed_length_basis(hmm.observations, 3, True)
    do_test_hmm_learning(
        hmm, learn, horizon, basis=basis,
        tau=tau, probabilistic=True, rank_tol=rank_tol)

    # results = []
    # for p in [True, False]:
    #     for estimator in ['prefix', 'substring']:
    #         for tau in [0.0001, 0.001, 0.01, 0.1, 1.0]:
    #             error = do_test_hmm_learning(
    #                 hmm, learn, horizon, basis=basis,
    #                 tau=tau, probabilistic=p, rank_tol=rank_tol)
    #             results.append(
    #                 dict(tau=tau, p=p, estimator=estimator, error=error))
    #             print results[-1]
    # print "Best: ", min(results, key=lambda x: x['error'])

    rr_hmm = reduced_rank_hmm()

    basis = fixed_length_basis(rr_hmm.observations, 3, False)
    do_test_hmm_learning(hmm, learn, 3, basis=basis, tau=0.001)

    basis = fixed_length_basis(rr_hmm.observations, 3, True)
    do_test_hmm_learning(hmm, learn, 3, basis=basis, tau=0.001)


@pytest.mark.xfail(reason="Incomplete implementation.")
def test_cts_sa():
    T = normalize(10 * np.eye(4) + np.ones((4, 4)), ord=1)
    O = [stats.multivariate_normal(np.ones(2), cov=0.1*np.eye(2)),
         stats.multivariate_normal(-np.ones(2), cov=0.1*np.eye(2)),
         stats.multivariate_normal(np.array([1, -1]), cov=0.1*np.eye(2)),
         stats.multivariate_normal(np.array([-1, 1]), cov=0.1*np.eye(2))]

    seed = 10
    np.random.seed(seed)

    init_dist = normalize([10, 1, 1, 1], ord=1)
    cts_hmm = ContinuousHMM(init_dist, T, O)

    n_train_samples = 1000
    horizon = 10

    eps = cts_hmm.sample_episodes(n_train_samples, horizon=horizon)
    all_data = np.array([s for t in eps for s in t])

    n_centers = 100
    cov = np.eye(cts_hmm.obs_dim)
    cov_inv = np.linalg.inv(cov)
    normal = stats.multivariate_normal(mean=np.zeros(cts_hmm.obs_dim), cov=cov)

    def kernel(x):
        return normal.pdf(x)

    def kernel_gradient(x):
        return cov_inv.dot(-x)

    kernel_centers = all_data[
        np.random.choice(len(all_data), size=n_centers, replace=False), :]
    lmbda = 0.5

    kernel_info = KernelInfo(kernel, kernel_centers, kernel_gradient, lmbda)
    sa = SpectralKernelSA(cts_hmm.n_states, kernel_info)
    sa.fit(eps)
    sa.reset()
    prediction = sa.predict()
    return sa, prediction


def test_mixture_seq_gen():
    n_components = 3
    n_states = 4
    n_obs = 5
    o_density, t_density = 0.5, 0.5

    seed = 10
    np.random.seed(seed)

    pfas = [
        make_pautomac_like(
            'hmm', n_states, n_obs, o_density, t_density, halts=False)
        for i in range(n_components)]

    coefficients = normalize([1.0] * n_components, ord=1)
    mixture_pfa = MixtureSeqGen(coefficients, pfas)

    # test that sampling doesn't crash
    mixture_pfa.sample_episodes(10, horizon=3)

    mixture_pfa.reset()

    obs_probs = [
        mixture_pfa.cond_obs_prob(o) for o in range(n_obs)]
    assert np.isclose(sum(obs_probs), 1)

    mixture_pfa.update(max(list(range(n_obs)), key=obs_probs.__getitem__))

    obs_probs = [
        mixture_pfa.cond_obs_prob(o) for o in range(n_obs)]
    assert np.isclose(sum(obs_probs), 1)

    mixture_pfa.update(max(list(range(n_obs)), key=obs_probs.__getitem__))

    obs_probs = [
        mixture_pfa.cond_obs_prob(o) for o in range(n_obs)]
    assert np.isclose(sum(obs_probs), 1)

    prefix_probs = [
        mixture_pfa.prefix_prob(string)
        for string in product(*[list(range(n_obs))]*3)]
    assert np.isclose(sum(prefix_probs), 1)

    mixture_pfa.reset()

    prefix_probs = [
        mixture_pfa.prefix_prob(string)
        for string in product(*[list(range(n_obs))]*3)]
    assert np.isclose(sum(prefix_probs), 1)


def test_modified_hmm():
    n_states = 4
    n_obs = 5
    o_density, t_density = 0.5, 0.5
    n_samples = 20

    seed = 10
    np.random.seed(seed)

    hmms = [
        make_pautomac_like(
            'hmm', n_states, n_obs, o_density, t_density, halts=True)
        for i in range(10)]

    for hmm in hmms:
        modified_hmm = hmm.modified()

        samples = hmm.sample_episodes(n_samples)
        for sample in samples:
            for i in range(len(sample)):
                prefix = sample[:i]
                normal_prob = hmm.prefix_prob(prefix)
                modified_prob = modified_hmm.prefix_prob(prefix)
                assert np.isclose(normal_prob, modified_prob)

                normal_prob = hmm.string_prob(prefix)
                modified_prob = modified_hmm.prefix_prob(prefix + [n_obs])
                assert np.isclose(normal_prob, modified_prob)


def test_markov_chain():
    seed = 10
    np.random.seed(seed)

    T = np.array([[0.9, 0.1], [0.3, 0.7]])
    init_dist = np.array([0.2, 0.8])
    horizon = 3
    mc = MarkovChain(init_dist, T)
    n_samples = 1000

    samples = mc.sample_episodes(n_samples, horizon=horizon)

    empirical_probs = defaultdict(int)
    for s in samples:
        empirical_probs[tuple(s)] += 1.0 / n_samples

    for seq, prob in six.iteritems(empirical_probs):
        reference = init_dist[seq[0]]
        for i in range(horizon-1):
            reference *= T[seq[i], seq[i+1]]

        model_guess = mc.prefix_prob(seq, log=False)
        assert np.isclose(reference, model_guess)

        # ``prob`` should be approximately normally distributed.
        std = np.sqrt(reference * (1 - reference) / n_samples)
        assert np.isclose(reference, prob, atol=4*std)


@pytest.mark.parametrize('allow_empty', [True, False])
def test_markov_chain_halt(allow_empty):
    seed = 10
    np.random.seed(seed)

    init_dist = np.array([0.2, 0.8])
    T = np.array([[0.9, 0.1], [0.3, 0.7]])
    stop_prob = np.array([0.5, 0.5])

    mc = MarkovChain(init_dist, T, stop_prob)
    if not allow_empty:
        mc = AdjustedMarkovChain(init_dist, T, stop_prob)

    T = np.diag(1 - stop_prob).dot(T)

    n_samples = 1000
    threshold = 0.5 / n_samples
    samples = mc.sample_episodes(n_samples)

    if not allow_empty:
        assert min(len(seq) for seq in samples) >= 1

    empirical_probs = defaultdict(int)
    for s in samples:
        empirical_probs[tuple(s)] += 1.0 / n_samples

    n_tested = 0
    n_ignored = 0
    for seq, prob in six.iteritems(empirical_probs):
        reference = mc.string_prob(seq)

        if allow_empty:
            s = init_dist.copy()
            for symbol in seq:
                s = s[symbol] * T[symbol, :]
            by_hand = s.dot(stop_prob)
        else:
            by_hand = np.log(init_dist[seq[0]])
            for i in range(len(seq)-1):
                by_hand += np.log(T[seq[i], seq[i+1]])
            by_hand += np.log(stop_prob[seq[-1]])
            by_hand = np.exp(by_hand)
        assert np.isclose(by_hand, reference)

        if reference > threshold:
            std = np.sqrt(reference * (1 - reference) / n_samples)
            assert np.isclose(reference, prob, atol=4*std), (
                "Disagreement for sample: %s" % str(seq))
            n_tested += 1
        else:
            n_ignored += 1

    print(("Num tested: %s, " % n_tested))
    print(("Num ignored: %s, " % n_ignored))
    print(("Num unique: %s, " % len(set(tuple(s) for s in empirical_probs))))


@pytest.mark.parametrize('from_dist', [True, False])
def test_markov_chain_learn(from_dist):
    seed = 10
    np.random.seed(seed)

    init_dist = np.array([0.2, 0.8])
    T = np.array([[0.9, 0.1], [0.3, 0.7]])
    stop_prob = np.array([0.2, 0.2])

    mc = AdjustedMarkovChain(init_dist, T, stop_prob)

    n_samples = 1000
    samples = mc.sample_episodes(n_samples)

    words = list(set(tuple(s) for s in samples))
    true_dist = np.array([mc.string_prob(w) for w in words])

    if from_dist:
        counts = defaultdict(int)
        for s in samples:
            counts[tuple(s)] += 1.0
        dist = np.array([counts[tuple(w)]/n_samples for w in words])
        learned_mc = AdjustedMarkovChain.from_distribution(
            dist, words, learn_halt=True, n_symbols=2)
    else:
        learned_mc = AdjustedMarkovChain.from_sequences(
            samples, learn_halt=True, n_symbols=2)
    learned_dist = np.array([learned_mc.string_prob(w) for w in words])

    error = rmse(true_dist, learned_dist)
    print(("RMSE: %s" % error))
    assert error < 0.001


@pytest.mark.skipif(True, reason="Incomplete implementation.")
def test_gmm_hmm_pendigits():
    import matplotlib.pyplot as plt
    from spectral_dagger.datasets import pendigits

    use_digits = [0]
    difference = True
    simplify = True

    max_data_points = None

    print("Loading data.")
    data, labels = pendigits.get_data(difference=difference, use_digits=use_digits, simplify=5, sample_every=3)
    labels = [l for ll in labels[:10] for l in ll]
    data = [d for dd in data[:10] for d in dd]

    if max_data_points is None:
        max_data_points = len(data)
    perm = np.random.permutation(len(data))[:max_data_points]

    labels = [labels[i] for i in perm]
    data = [data[i] for i in perm]
    print(("Amount of data: ", len(data)))

    pct_test = 0.2
    n_train = int((1-pct_test) * len(data))

    test_data = data[n_train:]
    test_labels = labels[n_train:]

    data = data[:n_train]
    labels = labels[:n_train]

    n_components = 1
    n_dim = 2

    for n_states in [10]:
        print(("Training with {0} states.".format(n_states)))
        gmm_hmm = GMMHMM(
            n_states=n_states, n_components=n_components, n_dim=n_dim,
            max_iter=1000, thresh=1e-4, verbose=0, cov_type='full',
            directory="temp", random_state=None, careful=False, left_to_right=True, n_restarts=5)

        gmm_hmm.fit(data)

        print(gmm_hmm.mean_log_likelihood(test_data))
        print(gmm_hmm.RMSE(test_data))


def test_gmm_hmm_simple():
    n_states = 10
    n_components = 2
    n_dim = 5

    n_train = 100
    n_test = 100

    horizon = 5

    n_restarts = 5

    print("Generating random ground truth model.")
    ground_truth = GMMHMM(n_states, n_components, n_dim)
    ground_truth._random_init()

    print("Generating data from ground truth.")
    train = ground_truth.sample_episodes(n_train, horizon=horizon)
    test = ground_truth.sample_episodes(n_test, horizon=horizon)

    print("Fitting model.")
    model = GMMHMM(n_states, n_components, n_dim, n_restarts=n_restarts, verbose=False)
    model.fit(train)
    print("Done.")

    print(("GROUND TRUTH MEAN LL: {0}.".format(ground_truth.mean_log_likelihood(test))))
    print(("GROUND TRUTH RMSE: {0}.".format(ground_truth.RMSE(test))))

    print(("MODEL MEAN LL: {0}.".format(model.mean_log_likelihood(test))))
    print(("MODEL RMSE: {0}.".format(model.RMSE(test))))
