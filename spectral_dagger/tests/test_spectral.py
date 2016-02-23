import pytest
import numpy as np
from scipy import stats

from spectral_dagger import sample_episodes
from spectral_dagger.spectral import SpectralPSR, CompressedPSR
from spectral_dagger.spectral import top_k_basis
from spectral_dagger.spectral import fixed_length_basis
from spectral_dagger.hmm import HMM, ContinuousHMM
from spectral_dagger.utils.math import normalize


@pytest.fixture
def simple_hmm():
    O = normalize([[1, 1], [.1, 1]], ord=1)
    T = normalize([[8, 2], [2, 8]], ord=1)

    init_dist = normalize([1, 1], ord=1)

    return HMM(T, O, init_dist)


@pytest.fixture
def reduced_rank_hmm():
    seed = np.random.randint(333)
    rng = np.random.RandomState(seed)

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

    O = np.eye(n)

    n_states = n
    n_obs = n

    return HMM(T, O, init_dist=normalize(np.ones(n), ord=1))


@pytest.mark.parametrize(
    'estimator', ['string', 'prefix', 'substring'])
def do_test_spectral_hmm(
        simple_hmm, estimator, n_samples=4000, horizon=3, m=None, basis=None):

    samples = sample_episodes(n_samples, simple_hmm, horizon=horizon)
    psr = SpectralPSR(simple_hmm.observations)

    if m is None:
        m = simple_hmm.n_states

    if basis is None:
        basis = top_k_basis(samples, np.inf, estimator)

    psr.fit(samples, m, estimator, basis=basis)

    test_seqs = [[0], [1], [0, 0], [0, 1], [1, 0], [1, 1]]

    print "*" * 20
    for seq in test_seqs:
        ground_truth = simple_hmm.get_seq_prob(seq)
        pred = psr.get_prefix_prob(seq)
        print("Seq: ", seq)
        print("Ground truth: %f" % ground_truth)
        print("Prediction: %f" % pred)
        assert np.isclose(ground_truth, pred, atol=0.2, rtol=0.0)


def do_test_compressed_hmm(
        simple_hmm, n_samples=100, horizon=3, m=None, basis=None):

    samples = sample_episodes(n_samples, simple_hmm, horizon=horizon)
    comp_psr = CompressedPSR(simple_hmm.observations)

    if m is None:
        m = simple_hmm.n_states

    if basis is None:
        basis = top_k_basis(samples, np.inf, 'prefix')

    comp_psr.model_rng = np.random.RandomState(10)
    comp_psr.fit(samples, m, basis=basis)

    test_seqs = [[0], [1], [0, 0], [0, 1], [1, 0], [1, 1]]

    print "*" * 20
    for seq in test_seqs:
        ground_truth = simple_hmm.get_seq_prob(seq)
        pred = comp_psr.get_prefix_prob(seq)
        print("Seq: ", seq)
        print("Ground truth: %f" % ground_truth)
        print("Prediction: %f" % pred)

        if n_samples >= 4000:
            assert np.isclose(ground_truth, pred, atol=0.2, rtol=0.0)


def test_psr():
    # Test SpectralPSR
    hmm = simple_hmm()
    dimension = 6

    do_test_spectral_hmm(hmm, 'string', m=dimension)
    do_test_spectral_hmm(hmm, 'prefix', m=dimension)
    do_test_spectral_hmm(hmm, 'substring', m=dimension)

    rr_hmm = reduced_rank_hmm()
    dimension = np.linalg.matrix_rank(rr_hmm.T)

    basis = fixed_length_basis(rr_hmm.observations, 2, False)
    do_test_spectral_hmm(rr_hmm, 'prefix', m=dimension, basis=basis, horizon=5)

    basis = fixed_length_basis(rr_hmm.observations, 1, False)
    do_test_spectral_hmm(rr_hmm, 'prefix', m=dimension, basis=basis, horizon=3)

    basis = fixed_length_basis(rr_hmm.observations, 2, True)
    do_test_spectral_hmm(rr_hmm, 'prefix', m=dimension, basis=basis, horizon=5)

    basis = fixed_length_basis(rr_hmm.observations, 1, True)
    do_test_spectral_hmm(rr_hmm, 'prefix', m=dimension, basis=basis, horizon=3)

    # Test CompressedPSR
    hmm = simple_hmm()
    dimension = 6

    do_test_compressed_hmm(hmm, m=dimension)

    rr_hmm = reduced_rank_hmm()
    dimension = np.linalg.matrix_rank(rr_hmm.T)

    basis = fixed_length_basis(rr_hmm.observations, 2, False)
    do_test_compressed_hmm(rr_hmm, m=dimension, basis=basis, horizon=5)

    basis = fixed_length_basis(rr_hmm.observations, 1, False)
    do_test_compressed_hmm(rr_hmm, m=dimension, basis=basis, horizon=3)

    basis = fixed_length_basis(rr_hmm.observations, 2, True)
    do_test_compressed_hmm(rr_hmm, m=dimension, basis=basis, horizon=5)

    basis = fixed_length_basis(rr_hmm.observations, 1, True)
    do_test_compressed_hmm(rr_hmm, m=dimension, basis=basis, horizon=3)


def test_cts_psr():
    T = normalize([[8, 2], [2, 8]], ord=1)
    O = [
        stats.multivariate_normal(np.ones(2)),
        stats.multivariate_normal(-np.ones(2))]

    init_dist = normalize([1, 1], ord=1)
    cts_hmm = ContinuousHMM(T, O, init_dist)
    eps = sample_episodes(10, cts_hmm, horizon=4)
    print eps

if __name__ == "__main__":
    test_cts_psr()
