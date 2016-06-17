import numpy as np

from spectral_dagger import get_model_rng
from spectral_dagger.utils import normalize
from spectral_dagger.sequence import ProbabilisticAutomaton


class HMM(ProbabilisticAutomaton):

    def __init__(self, T, O, init_dist=None, stop_prob=None):
        """ A Hidden Markov Model with discrete outputs.

        Parameters
        ----------
        T: ndarray
            A |states| x |states| matrix. Entry (i, j) gives
            the probability of moving from state i to state j, so each row of
            T must be a probability distribution.
        O: ndarray
            A |states| x |observations| matrix. Entry (i, j)
            gives the probability of emitting observation j given that the HMM
            is in state i, so each row of O must be a probablilty distribution.
        init_dist: ndarray
            A |states| vector specifying the initial state distribution.
            Defaults to a uniform distribution.
        stop_prob: ndarray
            A |states| vector specifying the prob of halting in each state.
        states: iterable (optional)
            The states that can be occupied by the HMM.
        observations: iterable (optional)
            The observations that can be emitted by the HMM.

        """
        self.states = range(T.shape[0])
        self.observations = range(O.shape[1])

        n_states = self.n_states = len(self.states)
        n_obs = self.n_observations = len(self.observations)

        self._T = T.copy()
        self._T.flags.writeable = False

        assert np.allclose(np.sum(self._T, axis=1), 1.0)
        assert np.all(self._T >= 0) and np.all(self._T <= 1)
        assert self._T.shape == (n_states, n_states)

        self._O = O.copy()
        self._O.flags.writeable = False

        assert np.allclose(np.sum(self._O, axis=1), 1.0)
        assert np.all(self._O >= 0) and np.all(self._O <= 1)
        assert self._O.shape == (n_states, n_obs)

        if init_dist is None:
            init_dist = np.ones(n_states) / float(n_states)
        self.init_dist = init_dist.copy()
        self.init_dist.flags.writeable = False
        assert(self.init_dist.size == n_states)
        assert(np.isclose(sum(self.init_dist), 1.0))
        assert np.all(self.init_dist >= 0) and np.all(self.init_dist <= 1)

        if stop_prob is None:
            stop_prob = np.zeros(n_states)
        self.stop_prob = stop_prob.copy()
        self.stop_prob.flags.writeable = False
        assert(self.stop_prob.size == n_states)
        assert np.all(self.stop_prob >= 0) and np.all(self.stop_prob <= 1)

        halt_correction = np.diag(1 - self.stop_prob)

        B_o = {o: np.diag(self._O[:, o]).dot(self._T).dot(halt_correction)
               for o in self.observations}
        super(HMM, self).__init__(
            self.init_dist, self.stop_prob, B_o, estimator='string')
        assert(is_hmm(self.b_0, self.b_inf_string, self.B_o))

    def __str__(self):
        return "<HMM. n_obs: %d, n_states: %d>" % (
            self.n_observations, self.n_states)

    @property
    def T(self):
        return self._T

    @property
    def O(self):
        return self._O


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


def dummy_hmm(n_states):
    """ Create a degenerate HMM. """

    n_obs = n_states

    O = np.eye(n_obs)
    T = np.eye(n_states)

    init_dist = normalize(np.ones(n_states), ord=1, conservative=True)

    hmm = HMM(T, O, init_dist)

    return hmm


def bernoulli_hmm(n_states, n_obs, model_rng=None):
    """ Create an HMM with operators chosen from Bernoulii distributions. """

    model_rng = get_model_rng() if model_rng is None else model_rng

    O = model_rng.binomial(1, 0.5, (n_states, n_obs))
    for row in O:
        if sum(row) == 0:
            row[:] = 1.0
    O = normalize(O, ord=1, conservative=True)

    T = model_rng.binomial(1, 0.5, (n_states, n_states))
    for row in T:
        if sum(row) == 0:
            row[:] = 1.0
    T = normalize(T, ord=1, conservative=True)

    init_dist = normalize(np.ones(n_states), ord=1, conservative=True)

    hmm = HMM(T, O, init_dist)

    return hmm
