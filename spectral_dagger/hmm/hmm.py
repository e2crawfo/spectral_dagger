import numpy as np
import time
from spectral_dagger.utils.math import normalize, sample_multinomial
from spectral_dagger.utils.math import default_rng
from spectral_dagger.spectral import PredictiveStateRep, fixed_length_basis


class HMM(object):

    def __init__(
            self, observations, states, T, O, init_dist=None, rng=None):
        """
        Parameters
        ----------
        observations: list
          The set of observations available.
        states: list
          The state space of the HMM.
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

        """
        self.observations = observations
        self.states = states

        self._O = O.copy()
        self._O.flags.writeable = False

        assert np.allclose(np.sum(self._O, axis=1), 1.0)
        assert np.all(self._O >= 0) and np.all(self._O <= 1)
        assert self._O.shape == (len(states), len(observations))

        self._T = T.copy()
        self._T.flags.writeable = False

        assert np.allclose(np.sum(self._T, axis=1), 1.0)
        assert np.all(self._T >= 0) and np.all(self._T <= 1)
        assert self._T.shape == (len(states), len(states))

        if init_dist is None:
            init_dist = np.ones(self.states) / self.states

        self.init_dist = init_dist.copy()
        assert(self.init_dist.size == len(self.states))
        assert(np.allclose(sum(init_dist), 1.0))

        self.rng = default_rng(rng)

        self.reset()

    @property
    def name(self):
        return "HMM"

    def __str__(self):
        return "%s. Current state: %s" % (
            self.name, str(self.current_state))

    def reset(self, init_dist=None):
        """
        Resets the state of the HMM.

        Parameters
        ----------
        state: State or int or ndarray or list
          If state is a State or int, sets the current state accordingly.
          Otherwise it must be all positive, sum to 1, and have length equal
          to the number of states in the MDP. The state is sampled from the
          induced distribution.

        """
        if init_dist is not None:
            raise ValueError("`init_dist` parameter must be None for HMMs.")

        self._current_state = self.states[
            sample_multinomial(self.init_dist, self.rng)]

    def sample_step(self):
        """ Returns the resulting observation. """

        obs = self.observations[
            sample_multinomial(self.O[self.current_state], self.rng)]

        self._current_state = self.states[
            sample_multinomial(self.T[self.current_state], self.rng)]

        return obs

    @property
    def n_observations(self):
        return len(self.observations)

    @property
    def n_states(self):
        return len(self.states)

    @property
    def current_state(self):
        return self._current_state

    @property
    def T(self):
        return self._T

    @property
    def O(self):
        return self._O

    def get_obs_prob(self, o):
        """ Get probability of observing obs given the current state. """
        return self._O[self._current_state, o]

    def get_seq_prob(self, seq):
        """ Get probability of observing `seq`.

        Disregards current internal state.

        """
        if len(seq) == 0:
            return 1.0

        s = self.init_dist.copy()

        for o in seq:
            s = s.dot(np.diag(self._O[:, o])).dot(self._T)

        return s.sum()

    def get_state_dist(self, t):
        """ Get state distribution after `t` steps. """
        s = self.init_dist.copy()

        for i in range(1, t):
            s = s.dot(self._T)

        return s

    def get_delayed_seq_prob(self, seq, t):
        """ Get probability of observing `seq` at a delay of `t`.

        get_delayed_seq_prob(seq, 0) is equivalent to get_seq_prob(seq).

        Disregards current internal state.

        """
        s = self.init_dist.copy()

        for i in range(t):
            s = s.dot(self._T)

        for o in seq:
            s = s.dot(np.diag(self._O[:, o])).dot(self._T)

        return s.sum()

    def get_subsequence_expectation(self, subseq, length):
        """ Get expected number of occurrences of `subseq` as subsequence.

        Assumes strings are of length `length`.

        Parameters
        ----------
        subseq: list/tuple/string
            Sequence of obervations.
        length: int > 0
            The length of sequences to consider.

        """
        if len(subseq) == 0:
            return length + 1

        reverse_seq = list(subseq[:-1])
        reverse_seq.reverse()

        # Compute vector of probability of sequence starting from each state.
        seq_given_state = self._O[:, subseq[-1]].copy()

        for obs in reverse_seq:
            seq_given_state = self._T.dot(seq_given_state)
            seq_given_state *= self._O[:, obs]

        s = self.init_dist.copy()

        prob = 0.0
        for i in range(length - len(subseq) + 1):
            prob += s.dot(seq_given_state)
            s = s.dot(self._T)

        return prob

    def sample_trajectory(self, horizon, reset=True, display=False):
        if reset:
            self.reset()

        trajectory = []

        if display:
            print "*" * 80

        for i in range(horizon):
            if display:
                print str(self)

            o = self.sample_step()
            trajectory.append(o)

            if display:
                print o
                time.sleep(0.3)

        if display:
            print str(self)

        return trajectory

    def to_psr(self):
        B_o = {}
        for o in self.observations:
            B_o[o] = self.T.dot(np.diag(self.O[:, o]))

        psr = PredictiveStateRep(
            b_0=self.init_dist, b_inf=np.ones(self.n_states), B_o=B_o)

        return psr


def dummy_hmm(n_states):
    """ Create a degenerate HMM. """

    n_obs = n_states

    observations = range(n_obs)
    states = range(n_states)

    O = np.eye(n_states)
    T = np.eye(n_states)

    init_dist = normalize(np.ones(n_states), ord=1, conservative=True)

    hmm = HMM(observations, states, T, O, init_dist)

    return hmm


def bernoulli_hmm(n_states, n_obs, model_rng=None):
    """ Create an HMM with operators chosen from Bernoulii distributions. """

    if model_rng is None:
        model_rng = default_rng(model_rng)

    observations = range(n_obs)
    states = range(n_states)

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

    hmm = HMM(observations, states, T, O, init_dist)

    return hmm


def test_hmm():
    observations = [0, 1]
    states = [0, 1]

    O = normalize(np.array([[1, 0], [0, 1]]), ord=1)
    T = normalize(np.array([[2, 1.], [3, 1]]), ord=1)

    init_dist = normalize(np.array([0.5, 0.5]), ord=1)

    hmm = HMM(observations, states, T, O, init_dist)
    print(hmm.sample_trajectory(10))


if __name__ == "__main__":
    test_hmm()
