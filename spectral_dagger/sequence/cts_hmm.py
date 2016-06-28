import numpy as np

from spectral_dagger import Space, Environment
from spectral_dagger.utils.math import sample_multinomial


class ContinuousHMM(Environment):

    def __init__(self, init_dist, T, O, states=None):
        """ A Hidden Markov Model with continuous observations.

        Parameters
        ----------
        init_dist: ndarray
            A |states| vector specifying the initial state distribution.
            Defaults to a uniform distribution.
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
        states: iterable (optional)
            The state space of the HMM.

        """
        self._states = range(T.shape[0]) if states is None else states
        n_states = len(self._states)

        self.init_dist = init_dist.copy()
        assert(self.init_dist.size == n_states)
        assert(np.allclose(sum(init_dist), 1.0))

        self._T = T.copy()
        self._T.flags.writeable = False

        assert np.allclose(np.sum(self._T, axis=1), 1.0)
        assert np.all(self._T >= 0) and np.all(self._T <= 1)
        assert self._T.shape == (n_states, n_states)

        self._O = O

        assert len(O) == n_states
        for o in O:
            assert hasattr(o, 'rvs')
            assert hasattr(o, 'pdf')

        self.reset()

    @property
    def name(self):
        return "ContinuousHMM"

    @property
    def action_space(self):
        return None

    @property
    def observation_space(self):
        return Space([-np.inf, np.inf] * len(self._O), "ObsSpace")

    def has_reward(self):
        return False

    def has_terminal_states(self):
        return False

    def in_terminal_state(self):
        return False

    @property
    def n_states(self):
        return len(self._states)

    @property
    def obs_dim(self):
        return self._O[0].rvs().size

    def reset(self, initial=None):
        self._state = self._states[
            sample_multinomial(self.init_dist, self.run_rng)]

    def step(self):
        """ Returns the resulting observation. """
        o = self._O[self._state].rvs(random_state=self.run_rng)
        self._state = self._states[
            sample_multinomial(self._T[self._state], self.run_rng)]

        return o

    def get_obs_prob(self, o):
        """ Get probability of observing o given the current state. """
        return self._O[self._state].pdf(o)

    def get_obs_probs(self, o):
        """ Get vector containing probs of observing obs given each state. """
        return np.array(
            [self._O[s].pdf(o) for s in self._states])
