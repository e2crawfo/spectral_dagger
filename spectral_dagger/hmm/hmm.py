import numpy as np
import time
from spectral_dagger.utils.math import normalize


class HMM(object):

    def __init__(
            self, observations, states, T, O, init_dist=None):
        """
        Parameters
        ----------
        observations: list
          The set of observations available.
        states: list
          The state space of the POMDP.
        T: ndarray
          A |states| x |states|  matrix. Entry (i, j) gives
          the probability of moving from state i to state j, so each row of
          T gives a probability distribution.
        O: ndarray
          A |states| x |observations| matrix. Entry (i, j)
          gives the probability of emitting observation j given that the HMM
          is in state i.
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

        self.reset()

    @property
    def name(self):
        return "HMM"

    def __str__(self):
        return "%s. Current state: %s" % (
            self.name, str(self.current_state))

    def reset(self, init_dist=None):
        """
        Resets the state of the POMDP.

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

        sample = np.random.multinomial(1, self.init_dist)
        self._current_state = self.states[np.where(sample > 0)[0][0]]


    def sample_step(self):
        """
        Returns the resulting observation.
        """
        sample = np.random.multinomial(1, self.O[self.current_state])
        obs = self.observations[np.where(sample > 0)[0][0]]

        sample = np.random.multinomial(1, self.T[self.current_state])
        self._current_state = self.states[np.where(sample > 0)[0][0]]

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

    def sample_trajectory(
            self, horizon, reset=True, display=False):

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
