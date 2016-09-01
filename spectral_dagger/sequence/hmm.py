import numpy as np
from sklearn.utils import check_random_state

from spectral_dagger.utils import normalize, indent
from spectral_dagger.sequence import ProbabilisticAutomaton


class HMM(ProbabilisticAutomaton):

    def __init__(self, init_dist, T, O, stop_prob=None):
        """ A Hidden Markov Model with discrete outputs.

        Parameters
        ----------
        init_dist: ndarray
            A |states| vector specifying the initial state distribution.
            Defaults to a uniform distribution.
        T: ndarray
            A |states| x |states| matrix. Entry (i, j) gives
            the probability of moving from state i to state j, so each row of
            T must be a probability distribution.
        O: ndarray
            A |states| x |observations| matrix. Entry (i, j)
            gives the probability of emitting observation j given that the HMM
            is in state i, so each row of O must be a probablilty distribution.
        stop_prob: ndarray
            A |states| vector specifying the prob of halting in each state.
        states: iterable (optional)
            The states that can be occupied by the HMM.
        observations: iterable (optional)
            The observations that can be emitted by the HMM.

        """
        self._states = range(T.shape[0])
        self._observations = range(O.shape[1])

        n_states = len(self._states)
        n_obs = len(self._observations)

        self._init_dist = np.array(init_dist).copy()
        self._init_dist.flags.writeable = False
        assert self._init_dist.size == n_states, str(self._init_dist)
        assert np.isclose(sum(self._init_dist), 1.0), str(self._init_dist)
        assert (
            np.all(self._init_dist >= 0) and
            np.all(self._init_dist <= 1)), str(self._init_dist)

        self._T = np.array(T).copy()
        self._T.flags.writeable = False

        assert self._T.shape == (n_states, n_states), str(self._T)
        assert np.allclose(np.sum(self._T, axis=1), 1.0), str(self._T)
        assert np.all(self._T >= 0) and np.all(self._T <= 1), str(self._T)

        self._O = np.array(O).copy()
        self._O.flags.writeable = False

        assert self._O.shape == (n_states, n_obs), str(self._O)
        assert np.allclose(np.sum(self._O, axis=1), 1.0), str(self._O)
        assert np.all(self._O >= 0) and np.all(self._O <= 1), str(self._O)

        if stop_prob is None:
            stop_prob = np.zeros(n_states)
        self._stop_prob = np.array(stop_prob).copy()
        self._stop_prob.flags.writeable = False

        assert self._stop_prob.size == n_states, str(self._stop_prob)
        assert (
            np.all(self._stop_prob >= 0) and
            np.all(self._stop_prob <= 1)), str(self._stop_prob)

        halt_correction = np.diag(1 - self._stop_prob)
        corrected_T = halt_correction.dot(self._T)

        B_o = {o: np.diag(self._O[:, o]).dot(corrected_T)
               for o in self._observations}
        super(HMM, self).__init__(
            self._init_dist, B_o, self._stop_prob, estimator='string')
        assert is_hmm(self.b_0, self.B_o, self.b_inf_string), str(self)

    def __str__(self):
        s = "HiddenMarkovModel\n"
        s += indent("n_states: %d\n" % self.n_states, 1)
        s += indent("n_symbols: %d\n" % self.n_observations, 1)
        s += indent("init_dist:\n", 1)
        s += indent(str(self.init_dist) + "\n", 2)
        s += indent("T:\n", 1)
        s += indent(str(self.T) + "\n", 2)
        s += indent("O:\n", 1)
        s += indent(str(self.O) + "\n", 2)
        s += indent("stop:\n", 1)
        s += indent(str(self.stop_prob) + "\n", 2)
        return s

    @property
    def init_dist(self):
        return self._init_dist

    @property
    def T(self):
        return self._T

    @property
    def O(self):
        return self._O

    @property
    def stop_prob(self):
        return self._stop_prob


def is_hmm(b_0, B_o, b_inf):
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


def dummy_hmm(n_states):
    """ Create a degenerate HMM. """

    n_obs = n_states

    O = np.eye(n_obs)
    T = np.eye(n_states)

    init_dist = normalize(np.ones(n_states), ord=1, conservative=True)

    return HMM(init_dist, T, O)


def bernoulli_hmm(n_states, n_obs, rng=None):
    """ Create an HMM with operators chosen from Bernoulii distributions. """
    rng = check_random_state(rng)
    O = rng.binomial(1, 0.5, (n_states, n_obs))
    for row in O:
        if sum(row) == 0:
            row[:] = 1.0
    O = normalize(O, ord=1, conservative=True)

    T = rng.binomial(1, 0.5, (n_states, n_states))
    for row in T:
        if sum(row) == 0:
            row[:] = 1.0
    T = normalize(T, ord=1, conservative=True)

    init_dist = normalize(np.ones(n_states), ord=1, conservative=True)

    return HMM(init_dist, T, O)
