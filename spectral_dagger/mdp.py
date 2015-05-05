import numpy as np
import time


class State(object):
    def __init__(self, id, dimension, name=""):
        self.id = id
        self.dimension = dimension
        self.ndim = 0
        self.name = name

    def as_vector(self):
        vec = np.array(np.zeros(self.dimension))
        vec[self.id] = 1.0
        return vec

    def get_id(self):
        return self.id

    def __str__(self):
        s = "<State id: %d, dim: %d" % (self.get_id(), self.dimension)
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


class Action(object):
    def __init__(self, id, name=""):
        self.id = id
        self.ndim = 0
        self.name = name

    def get_id(self):
        return self.id

    def __str__(self):
        s = "<Action id: %d" % self.get_id()
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


class MDP(object):

    def __init__(self, actions, states, T, R, gamma, initial_state=None):
        """
        Parameters
        ----------
        actions: list
          The set of actions available.
        states: list
          The state space of the MDP.
        T: ndarray
          An |actions| x |states| x |states|  matrix. Entry (a, i, j) gives
          the probability of moving from state i to state j given that action
          a taken.
        R: ndarray
          An |actions| x |states| matrix. Entry (a, i) gives the reward for
          ending up in state i given we took action a.
        gamma: float
          Discount factor.
        """

        self.actions = actions
        self.states = states
        self.initial_state = initial_state

        self._T = T
        self._R = R

        self.gamma = gamma

    @property
    def name(self):
        return "DefaultMDP"

    def __str__(self):
        return "%s. Current state: %s" % (
            self.name, str(self.current_state))

    def reset(self, state=None):
        """
        Resets the state of the MDP.

        Parameters
        ----------
        state: State or int or or None
          If state is a State or int, sets the current state accordingly.
          If None, states are chosen uniformly at random.
        """

        if state is None and self.initial_state:
            self.current_state = self.initial_state
        elif isinstance(state, State):
            self.current_state = state
        elif isinstance(state, int):
            self.current_state = self.states[state]
        else:
            self.current_state = np.random.choice(self.states)

    def execute_action(self, action):
        """
        Play the given action.

        Returns the next state and the reward.
        """

        if not isinstance(action, Action) and not isinstance(action, int):
            raise ValueError(
                "Action must either be an integer or an instance of the "
                "Action class. Got object of type %s instead." % type(action))

        if isinstance(action, int):
            action = self.actions[action]

        sample = np.random.multinomial(1, self._T[action, self.current_state])
        self.current_state = self.states[np.where(sample > 0)[0][0]]

        reward = self.get_reward(action, self.current_state)

        return self.current_state, reward

    @property
    def num_actions(self):
        return len(self.actions)

    @property
    def num_states(self):
        return len(self.states)

    @property
    def state(self):
        return self.current_state

    @property
    def T(self):
        return self._T.copy()

    @property
    def R(self):
        return self._R.copy()

    def get_reward(self, action, state):
        return self._R[action, state]

    def sample_trajectory(
            self, mdp_policy, horizon, reset=None,
            init=None, return_reward=True, display=False):

        if reset:
            if isinstance(init, np.ndarray) or isinstance(init, list):
                sample = np.random.multinomial(1, init)
                init = self.states[np.where(sample > 0)[0][0]]

            self.reset(init)

        mdp_policy.reset(self.state)

        trajectory = []

        if display:
            print "*" * 80

        for i in range(horizon):
            if display:
                print str(self)

            s = self.state

            a = mdp_policy.get_action()

            s_prime, r = self.execute_action(a)

            if return_reward:
                trajectory.append((s, a, r))
            else:
                trajectory.append((s, a))

            mdp_policy.update(a, s_prime, r)

            if display:
                print a
                print s_prime
                time.sleep(0.3)

        if display:
            print str(self)

        return trajectory


class MDPPolicy(object):
    """
    The most general MDPPolicy class.

    Parameters
    ----------
    pi: dict or callable
        A mapping from states to actions.
    """

    def __init__(self, pi):
        self.is_dict = hasattr(pi, '__getitem__')

        if not self.is_dict and not callable(pi):
            raise Exception(
                "pi must be either a dict or a callable.")

        self.pi = pi

    def reset(self, state):
        self.current_state = state

    def update(self, action, state, reward=None):
        self.current_state = state

    def get_action(self):
        if self.is_dict:
            return self.pi[self.current_state]
        else:
            return self.pi(self.current_state)


class UniformRandomPolicy(MDPPolicy):

    def __init__(self, mdp):
        self.actions = mdp.actions

    def get_action(self):
        return np.random.choice(self.actions)


class GreedyPolicy(MDPPolicy):

    def __init__(self, mdp, V):
        self.T = mdp.T
        self.R = mdp.R
        self.gamma = mdp.gamma
        self.actions = mdp.actions

        self.V = V.copy()

    def set_value(self, s, v):
        self.V[s] = v

    def get_action(self):
        T_s = self.T[:, self.current_state, :]

        return max(
            self.actions,
            key=lambda a: T_s[a, :].dot(self.R[a, :] + self.gamma * self.V))
