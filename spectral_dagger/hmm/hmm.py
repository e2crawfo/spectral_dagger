import numpy as np

from spectral_dagger import Environment, Space, sample_episodes, get_model_rng
from spectral_dagger.utils.math import normalize, sample_multinomial
from spectral_dagger.spectral import PredictiveStateRep


class HMM(Environment):

    def __init__(self, T, O, init_dist=None, states=None, observations=None):
        """ A Hidden Markov Model.

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
        states: iterable (optional)
            The states that can be occupied by the HMM.
        observations: iterable (optional)
            The observations that can be emitted by the HMM.

        """
        self.states = range(T.shape[0]) if states is None else states
        self.observations = (
            range(O.shape[1]) if observations is None else observations)

        n_states = len(self.states)
        n_obs = len(self.observations)

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
        assert(self.init_dist.size == n_states)
        assert(np.isclose(sum(init_dist), 1.0))

        self.reset()

    @property
    def name(self):
        return "HMM"

    def __str__(self):
        return "<%s. Current state: %s>" % (
            self.name, str(self.state))

    @property
    def action_space(self):
        return None

    @property
    def observation_space(self):
        return Space([set(self.observations)], "ObsSpace")

    def has_terminal_states(self):
        return False

    def in_terminal_state(self):
        return False

    def has_reward(self):
        return False

    @property
    def n_observations(self):
        return len(self.observations)

    @property
    def n_states(self):
        return len(self.states)

    def reset(self, init_dist=None):
        """
        Resets the state of the HMM.

        Parameters
        ----------
        init_dist: array-like (optional)
            A vector giving an initial distribution over hidden states.

        """
        if init_dist is None:
            init_dist = self.init_dist

        self._state = self.states[
            sample_multinomial(init_dist, self.rng)]

    def update(self):
        """ Returns the resulting observation. """

        o = self.observations[
            sample_multinomial(self.O[self.state], self.rng)]

        self._state = self.states[
            sample_multinomial(self.T[self.state], self.rng)]

        return o

    @property
    def current_state(self):
        return self._state

    @property
    def state(self):
        return self._state

    @property
    def T(self):
        return self._T

    @property
    def O(self):
        return self._O

    def get_obs_prob(self, o):
        """ Get probability of observing obs given the current state. """
        return self._O[self.state, o]

    def get_obs_probs(self, o):
        """ Get vector containing probs of observing obs given each state. """
        return self._O[:, o]

    def get_seq_prob(self, seq):
        """ Get probability of observing `seq`.

        Disregards current internal state.

        """
        if len(seq) == 0:
            return 1.0

        s = self.init_dist.copy()

        for o in seq:
            s = (s * self.get_obs_probs(o)).dot(self._T)

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
            s = (s.dot * self.get_obs_probs(o)).dot(self._T)

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
        seq_given_state = self.get_obs_probs(subseq[-1]).copy()

        for o in reverse_seq:
            seq_given_state = self._T.dot(seq_given_state)
            seq_given_state *= self.get_obs_probs(o)

        s = self.init_dist.copy()

        prob = 0.0
        for i in range(length - len(subseq) + 1):
            prob += s.dot(seq_given_state)
            s = s.dot(self._T)

        return prob

    def to_psr(self):
        B_o = {}
        for o in self.observations:
            B_o[o] = np.diag(self.get_obs_probs(o)).dot(self.T)

        psr = PredictiveStateRep(
            b_0=self.init_dist, b_inf=np.ones(self.n_states), B_o=B_o,
            estimator='prefix', can_terminate=False)

        return psr


class ContinuousHMM(HMM):

    def __init__(self, T, O, init_dist=None, states=None):
        """ A Hidden Markov Model with continuous observations.

        Parameters
        ----------
        T: ndarray
            A |states| x |states| matrix. Entry (i, j) gives
            the probability of moving from state i to state j, so each row of
            T must be a probability distribution.
        O: A list of distributions.
            A list of length |states|. The i-th element is a distribution (so
            has an ``rvs`` method, which takes an argument called
            ``random_state``, and a ``pdf`` method; basically, instances
            of scipy.stats.rv_frozen) which gives the probability density
            for observation emissions given the i-th state. All distributions
            should have the same dimensionality.
        init_dist: ndarray
            A |states| vector specifying the initial state distribution.
            Defaults to a uniform distribution.
        states: iterable (optional)
            The state space of the HMM.

        """
        self.states = range(T.shape[0]) if states is None else states
        n_states = len(self.states)

        self._T = T.copy()
        self._T.flags.writeable = False

        assert np.allclose(np.sum(self._T, axis=1), 1.0)
        assert np.all(self._T >= 0) and np.all(self._T <= 1)
        assert self._T.shape == (n_states, n_states)

        self._O = O

        assert len(self.O) == n_states
        for o in O:
            assert hasattr(o, 'rvs')
            assert hasattr(o, 'pdf')

        if init_dist is None:
            init_dist = np.ones(n_states) / float(n_states)

        self.init_dist = init_dist.copy()
        assert(self.init_dist.size == n_states)
        assert(np.allclose(sum(init_dist), 1.0))

        self.reset()

    @property
    def name(self):
        return "ContinuousHMM"

    @property
    def observation_space(self):
        return Space([-np.inf, np.inf] * len(self.O), "ObsSpace")

    def update(self):
        """ Returns the resulting observation. """
        o = self.O[self.state].rvs(random_state=self.rng)
        self._state = self.states[
            sample_multinomial(self.T[self.state], self.rng)]

        return o

    def get_obs_prob(self, o):
        """ Get probability of observing o given the current state. """
        return self.O[self.state].pdf(o)

    def get_obs_probs(self, o):
        """ Get vector containing probs of observing obs given each state. """
        return np.array(
            [self._O[s].pdf(o) for s in self.states])

    def to_psr(self):
        raise NotImplementedError()


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


def test_hmm():
    O = normalize(np.array([[1, 0], [0, 1]]), ord=1)
    T = normalize(np.array([[2, 1.], [3, 1]]), ord=1)

    init_dist = normalize(np.array([0.5, 0.5]), ord=1)

    hmm = HMM(T, O, init_dist)
    print(sample_episodes(10, hmm))


if __name__ == "__main__":
    test_hmm()
