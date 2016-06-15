import numpy as np
import six

from spectral_dagger import Environment, Space
from spectral_dagger.sequence import StochasticAutomaton
from spectral_dagger.utils import normalize
from spectral_dagger.utils import sample_multinomial


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


def is_hmm(b_0, b_inf, B_o):
    """ Check that b_0, b_inf, B_o form a Hidden Markov Model.

    Will only return True if it is an HMM in standard form
    (i.e. B_sigma = diag(O_sigma)T)

    ``b_inf`` is assumed to be normalization vector for strings
    (i.e. halting vector).

    """
    if not np.isclose(b_0.sum(), 1.0):
        return False

    if np.any(b_0 < 0) or np.any(b_0 > 1):
        return False

    if np.any(b_inf < 0) or np.any(b_inf > 1):
        return False

    for i in range(b_inf.size):
        first = True
        coefs = []

        for o in B_o:
            if first:
                first_row = B_o[o][i, :].copy()
                s = first_row.sum()

                if s > 0:
                    first_row /= s
                    first = False

                coefs.append(s)
            else:
                row = B_o[o][i, :].copy()
                s = row.sum()

                if s > 0:
                    row /= s

                    if not np.allclose(first_row, row):
                        return False

                coefs.append(s)

        if not np.isclose(sum(coefs) + b_inf[i], 1.0):
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
    pfa: StochasticAutomaton instance
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

    return StochasticAutomaton(
        b_0, b_inf_string, Bo_prime, estimator='string')


def perturb_pfa_multiplicative(pfa, std, rng=None):
    """ Generate a perturbed version of a PFA.

    Multiply each non-zero element by (1 + epsilon), where epsilon
    is Gaussian distributed.

    Parameters
    ----------
    pfa: StochasticAutomaton instance
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

    return StochasticAutomaton(
        b_0, b_inf_string, Bo_prime, estimator='string')


def perturb_pfa_bernoulli(pfa, p, increment=None, rng=None):
    """ Generate a perturbed version of a PFA using additive bernoulli noise.

    Parameters
    ----------
    pfa: StochasticAutomaton instance
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

    return StochasticAutomaton(
        b_0, b_inf_string, Bo_prime, estimator='string')


class PFASampler(Environment):
    """ An environment created from a probabilistic automaton. """

    def __init__(self, pa):
        self.observations = pa.observations
        self.can_terminate = pa.can_terminate
        self.terminal = False

        self.b_0 = pa.b_0.copy()

        self.B_o = {}
        for o in pa.B_o:
            self.B_o[o] = pa.B_o[o].copy()

        self.b_inf_string = pa.b_inf_string.copy()
        self.b_inf_prefix = pa.b_inf.copy()

        self.reset()

    @property
    def action_space(self):
        return None

    @property
    def observation_space(self):
        return Space(set(self.observations), "ObsSpace")

    @property
    def size(self):
        return self.b_0.size

    def in_terminal_state(self):
        return self.terminal

    def has_terminal_states(self):
        return self.can_terminate

    def has_reward(self):
        return False

    def lookahead(self):
        terminal_prob = self.b.dot(self.b_inf_string)
        self.terminal = self.rng.rand() < terminal_prob

        probs = np.array([
            self.b.dot(self.B_o[o]).dot(self.b_inf_prefix)
            for o in self.observations])

        # Normalize probs since we've already sampled whether to terminate.
        self.probs = probs / probs.sum()

    def reset(self, initial=None):
        self.b = self.b_0.copy()
        self.lookahead()

    def step(self):
        if self.terminal:
            return None

        sample = sample_multinomial(self.probs, self.rng)
        obs = self.observations[sample]

        numer = self.b.dot(self.B_o[obs])
        denom = numer.dot(self.b_inf_prefix)
        if np.isclose(denom, 0):
            self.b = np.zeros_like(self.b)
        else:
            self.b = numer / denom

        self.lookahead()

        return obs

    def to_sa(self):
        sa = StochasticAutomaton(
            b_0=self.b_0, b_inf=self.b_inf_prefix, B_o=self.B_o,
            estimator='prefix')

        return sa
