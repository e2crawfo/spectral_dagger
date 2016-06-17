import numpy as np
import six

from spectral_dagger.sequence import StochasticAutomaton
from spectral_dagger.utils import normalize


class ProbabilisticAutomaton(StochasticAutomaton):
    def __init__(self, b_0, b_inf, B_o, estimator):
        super(ProbabilisticAutomaton,
              self).__init__(b_0, b_inf, B_o, estimator)

        assert is_pfa(self.b_0, self.b_inf_string, self.B_o)

    def __str__(self):
        return ("<ProbabilisticAutomaton. "
                "n_obs: %d, n_states: %d>" % (self.n_observations, self.size))


def is_pfa(b_0, b_inf, B_o):
    """ Check that b_0, b_inf, B_o form a Probabilistic Finite Automaton.

    ``b_inf`` is assumed to be normalization vector for strings
    (i.e. halting vector).

    """
    B = sum(B_o.values())
    B_row_sums = B.sum(axis=1)
    return np.allclose(B_row_sums + b_inf, 1)


def is_dpfa(b_0, b_inf, B_o):
    """ Check that b_0, b_inf, B_o form a DPFA.
    DPFA stands for Deterministic Probabilistic Finite Automaton.

    ``b_inf`` is assumed to be normalization vector for strings
    (i.e. halting vector).

    """
    if np.count_nonzero(b_0) > 1:
        return False
    for o in B_o:
        for i in range(B_o[o].shape[0]):
            if np.count_nonzero(B_o[o][i, :]) > 1:
                return False
    return True


def normalize_pfa(b_0, b_inf, B_o):
    """ Return a version of arguments normalized to be a PFA. """

    assert (b_0 >= 0).all()
    assert (b_inf >= 0).all()

    for o, B in six.iteritems(B_o):
        assert (B >= 0).all()

    b_0 = normalize(b_0, ord=1)

    B = sum(B_o.values())
    norms = B.sum(axis=1) / (1 - b_inf)

    norms = norms.reshape(-1, 1)
    new_B_o = {}
    for o, Bo in six.iteritems(B_o):
        new_B_o[o] = Bo / norms
        new_B_o[o][np.isnan(new_B_o[o])] = 0.0

    assert is_pfa(b_0, b_inf, new_B_o)
    return b_0, b_inf, new_B_o


def perturb_pfa_additive(pfa, std, rng=None):
    """ Generate a perturbed version of a PFA using additive noise.

    Noise takes the form of a one-sided Gaussian.

    Parameters
    ----------
    pfa: ProbabilisticAutomaton instance
        The PFA to perturb.
    std: positive float
        Standard deviation of perturbation for the operators.

    """
    rng = rng if rng is not None else np.random.RandomState()

    # Only perturb locations that are already non-zero.
    Bo_prime = {}
    for o, b in six.iteritems(pfa.B_o):
        b_prime = b.copy()
        b_prime[b_prime > 0] += np.absolute(
            std * rng.randn(np.count_nonzero(b_prime)))
        Bo_prime[o] = b_prime

    b_0, b_inf_string, Bo_prime = normalize_pfa(
        pfa.b_0, pfa.b_inf_string, Bo_prime)

    return ProbabilisticAutomaton(
        b_0, b_inf_string, Bo_prime, estimator='string')


def perturb_pfa_multiplicative(pfa, std, rng=None):
    """ Generate a perturbed version of a PFA.

    Multiply each non-zero element by (1 + epsilon), where epsilon
    is Gaussian distributed.

    Parameters
    ----------
    pfa: ProbabilisticAutomaton instance
        The PFA to perturb.
    std: positive float
        Standard deviation of perturbation for the operators.

    """
    rng = rng if rng is not None else np.random.RandomState()

    # Only perturb locations that are already non-zero.
    Bo_prime = {}
    for o, b in six.iteritems(pfa.B_o):
        b_prime = b.copy()
        b_prime[b_prime > 0] *= 1 + std * rng.randn(np.count_nonzero(b_prime))
        Bo_prime[o] = np.absolute(b_prime)

    b_0, b_inf_string, Bo_prime = normalize_pfa(
        pfa.b_0, pfa.b_inf_string, Bo_prime)

    return ProbabilisticAutomaton(
        b_0, b_inf_string, Bo_prime, estimator='string')


def perturb_pfa_bernoulli(pfa, p, increment=None, rng=None):
    """ Generate a perturbed version of a PFA using additive bernoulli noise.

    Parameters
    ----------
    pfa: ProbabilisticAutomaton instance
        The PFA to perturb.
    p: positive float
        Probability parameter for Bernoulli's.
    increment: float
        The amount to increment when the Bernoulli is non-zero.

    """
    rng = rng if rng is not None else np.random.RandomState()
    increment = (
        increment if increment is not None
        else np.mean(np.mean(np.hstack(pfa.B_o.values()))))

    Bo_prime = {}
    for o, b in six.iteritems(pfa.B_o):
        b_prime = b.copy()
        b_prime += np.abs(increment * rng.binomial(1, p, size=b_prime.shape))
        Bo_prime[o] = b_prime

    b_0, b_inf_string, Bo_prime = normalize_pfa(
        pfa.b_0, pfa.b_inf_string, Bo_prime)

    return ProbabilisticAutomaton(
        b_0, b_inf_string, Bo_prime, estimator='string')
