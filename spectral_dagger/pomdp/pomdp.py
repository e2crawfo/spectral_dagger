import numpy as np

from spectral_dagger import Policy, Environment, Space
from spectral_dagger.mdp import MDP
from spectral_dagger.utils import sample_multinomial


class POMDP(Environment):

    def __init__(
            self, actions, observations, states,
            T, O, R, init_dist=None, gamma=1.0):
        """ A Partially Observable Markov Decision Process.

        Parameters
        ----------
        actions: list
            The list of actions.
        observations: list
            The list of observations.
        states: list
            The list of states.
        T: ndarray
            A |actions| x |states| x |states|  matrix. Entry (a, i, j) gives
            the probability of moving from state i to state j given that action
            a is taken.
        O: ndarray
            A |actions| x |states| x |observations| matrix. Entry (a, i, j)
            gives the probability of emitting observation j given that the
            POMDP is in state i and the last action was a.
        R: ndarray
            An |actions| x |states| x |states| matrix. Entry (a, i, j) gives
            the reward for transitioning from state i to state j with action a.
        init_dist: ndarray (optional)
            A |states| vector specifying the initial state distribution.
            Defaults to a uniform distribution.
        gamma: float (optional)
            Discount factor between 0 and 1 (inclusive).

        """
        self.actions = actions
        self.observations = observations
        self.states = states

        self._O = O.copy()
        self._O.flags.writeable = False

        assert all(np.sum(self._O, axis=2) == 1.0)
        assert all(self._O > 0) and all(self._O < 1)
        assert self._O.shape == (len(actions), len(states), len(observations))

        if init_dist is None:
            init_dist = np.ones(self.states) / self.states

        self.init_dist = init_dist.copy()
        assert(self.init_dist.size == len(self.states))
        assert(sum(init_dist) == 1)

        self.mdp = MDP(actions, states, T, R, gamma)

        self.reset()

    @property
    def name(self):
        return "DefaultPOMDP"

    def __str__(self):
        return "%s. Current state: %s" % (
            self.name, str(self.current_state))

    @property
    def n_actions(self):
        return len(self.actions)

    @property
    def n_observations(self):
        return len(self.observations)

    @property
    def n_states(self):
        return len(self.states)

    @property
    def action_space(self):
        return Space([set(self.actions)], "ActionSpace")

    @property
    def observation_space(self):
        return Space([set(self.observations)], "ObsSpace")

    def has_reward(self):
        return True

    def has_terminal_states(self):
        return (
            hasattr(self, 'terminal_states')
            and bool(self.terminal_states))

    def in_terminal_state(self):
        return (
            self.has_terminal_states()
            and self.current_state in self.terminal_states)

    def reset(self, init_dist):
        """ Resets the state of the POMDP.

        Parameters
        ----------
        init_dist: array-like (optional)
            A distribution to choose the initial state from.

        """
        if init_dist is None:
            init_dist = self.init_dist

        self.mdp.reset(init_dist)

    def step(self, action):
        """ Execute the given action. Returns new obs and reward. """

        state, reward = self.mdp.step(action)

        obs_dist = self.O[action, self.mdp.current_state]
        obs = sample_multinomial(obs_dist)

        return obs, reward

    @property
    def gamma(self):
        return self.mdp.gamma

    @property
    def current_state(self):
        return self.mdp.current_state

    @property
    def T(self):
        return self.mdp.T

    @property
    def O(self):
        return self._O

    @property
    def R(self):
        return self.mdp.R

    def get_expected_reward(self, a, s):
        return self.mdp.get_expected_reward(a, s)

    def get_reward(self, a, s, s_prime=None):
        return self.mdp.get_reward(a, s, s_prime)


class HistoryPolicy(Policy):
    """ A policy that chooses actions by looking at the full history. """

    def __init__(self, f):
        self.f = f
        self.history = []

    def reset(self, init_dist=None):
        self.history = []

    def update(self, observation, action, reward=None):
        self.history.append((action, observation, reward))

    def get_action(self):
        return self.f(self.history)


class BeliefStatePolicy(Policy):
    """ A policy that chooses actions by maintaining a belief state.

    The policy has a access to a model of the environment in the form
    of a POMDP, and can thus maintain a belief state. Accepts a function
    ``pi`` which maps belief states to actions.

    """
    def __init__(self, pomdp, pi=None):
        self.pomdp = pomdp
        self.b = None

        if pi is None:
            def f(b, rng=self.random_state):
                return rng.choice(pomdp.actions)

            pi = f

        assert callable(pi)
        self.pi = pi

    @property
    def action_space(self):
        return self.pomdp.action_space

    @property
    def observation_space(self):
        return self.pomdp.observation_space

    @property
    def belief_state(self):
        return self.b

    def reset(self, b=None):
        if b is None:
            b = self.pomdp.init_dist

        self.b = b

    def update(self, o, a, r=None):
        b_prime = self.b.dot(self.pomdp.T[a]) * self.pomdp.O[a, :, o]
        b_prime /= sum(b_prime)
        self.b = b_prime

    def get_action(self):
        return self.pi(self.b)
