import numpy as np
import time

from mdp import MDP


class Observation(object):
    def __init__(self, id, name=""):
        self.id = id
        self.ndim = 0
        self.name = name

    def get_id(self):
        return self.id

    def __str__(self):
        s = "<Observation id: %d" % self.get_id()
        if self.name:
            s += ", name: " + self.name
        return s + ">"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        """Override the default == behavior"""
        if isinstance(other, self.__class__):
            return self.get_id() == other.get_id()
        elif isinstance(other, int):
            return self.get_id() == other

        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, self.__class__) or isinstance(other, int):
            return not self.__eq__(other)

        return NotImplemented

    def __hash__(self):
        """Override default behaviour when used as key in dict"""
        return hash(self.get_id())

    def __index__(self):
        return self.get_id()

    def __int__(self):
        return self.get_id()


class POMDP(object):

    def __init__(
            self, actions, observations, states,
            T, O, R, init_dist=None, gamma=1.0):
        """
        Parameters
        ----------
        actions: list
          The set of actions available.
        observations: list
          The set of observations available.
        states: list
          The state space of the POMDP.
        T: ndarray
          An |actions| x |states| x |states|  matrix. Entry (a, i, j) gives
          the probability of moving from state i to state j given that action
          a taken.
        O: ndarray
          An |actions| x |states| x |observations| matrix. Entry (a, i, j)
          gives the probability of emitting observation j given that the POMDP
          is in state i and the last action was a.
        R: ndarray
          An |actions| x |states| matrix. Entry (a, i) gives the reward for
          ending up in state i given we took action a.
        init_dist: ndarray
          A |states| ndarray specifying the default state initiazation
          distribution. If none is provided, a uniform distribution is used.
        gamma: float
          Discount factor.
        """

        self.actions = actions
        self.observations = observations
        self.states = states

        self._T = T
        self._O = O
        self._R = R

        if init_dist is None:
            init_dist = np.ones(self.states) / self.states

        self.init_dist = init_dist

        self.mdp = MDP(actions, states, T, R, gamma)

    @property
    def name(self):
        return "DefaultPOMDP"

    def __str__(self):
        return "%s. Current state: %s" % (
            self.name, str(self.current_state))

    def reset(self, init_dist):
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

        if init_dist is None:
            init_dist = self.init_dist

        sample = np.random.multinomial(1, init_dist)
        self.mdp.reset(np.where(sample > 0)[0][0])

    def execute_action(self, action):
        """
        Play the given action.

        Returns the resulting observation and reward.
        """

        state, reward = self.mdp.execute_action(action)

        sample = np.random.multinomial(1, self.O[action, self.mdp.state])
        obs = Observation(np.where(sample > 0)[0][0])

        return obs, reward

    @property
    def num_actions(self):
        return len(self.actions)

    @property
    def num_observations(self):
        return len(self.observations)

    @property
    def num_states(self):
        return len(self.states)

    @property
    def state(self):
        return self.mdp.state

    @property
    def T(self):
        return self._T.copy()

    @property
    def O(self):
        return self._O.copy()

    @property
    def R(self):
        return self._R.copy()

    def get_reward(self, action, state):
        return self.mdp.get_reward(action, state)

    def sample_trajectory(
            self, pomdp_policy, horizon, reset=True,
            init_dist=None, return_reward=True, display=False):

        if reset:
            # init_dist = None means policy and environment have agreed on an
            # initial distribution before-hand which they both, in some sense,
            # have access to.
            #
            # TODO: all pomdp policies should take an initial distribution as a
            # in their reset functions. If they only work for fixed initial
            # distributions, then they should throw an exception if they get a
            # non-None distribution.

            self.reset(init_dist)
            pomdp_policy.reset(init_dist)

        trajectory = []

        if display:
            print "*" * 80

        for i in range(horizon):
            if display:
                print str(self)

            a = pomdp_policy.get_action()

            o, r = self.execute_action(a)

            if return_reward:
                trajectory.append((a, o, r))
            else:
                trajectory.append((a, o))

            pomdp_policy.update(a, o, r)

            if display:
                print a
                print o
                time.sleep(0.3)

        if display:
            print str(self)

        return trajectory


class POMDPPolicy(object):
    """
    A policy that operates on a POMDP.
    Uses an arbitrary function of the history  to choose actions.
    """

    def __init__(self, f):
        self.f = f
        self.history = []

    def reset(self, init_dist=None):
        self.history = []

    def update(self, action, observation, reward=None):
        self.history.append((action, observation))

    def get_action(self):
        return self.f(self.history)


class UniformRandomPolicy(POMDPPolicy):

    def __init__(self, pomdp):
        self.actions = pomdp.actions

    def get_action(self):
        return np.random.choice(self.actions)


class BeliefStatePolicy(POMDPPolicy):
    """
    A policy that has a access to a model of the environment in the form
    of a POMDP, and can thus maintain a belief state. Accepts a function
    pi which maps belief states to actions.
    """

    def __init__(self, pomdp, pi=None):
        self.pomdp = pomdp
        self.b = None

        if pi is None:
            def f():
                return np.random.choice(pomdp.actions)
            pi = f

        assert callable(pi)
        self.pi = pi

    @property
    def belief_state(self):
        return self.b

    def reset(self, b=None):
        if b is None:
            b = self.pomdp.init_dist

        self.b = b

    def update(self, a, o, r=None):
        b_prime = self.b.dot(self.pomdp.T[a]) * self.pomdp.O[a, :, o]
        b_prime /= sum(b_prime)
        self.b = b_prime

    def get_action(self):
        return self.pi(self.b)
