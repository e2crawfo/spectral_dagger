import numpy as np
import time

from spectral_dagger.mdp.policy import UniformRandomPolicy


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

    def __init__(
            self, actions, states, T, R, gamma,
            initial_state=None, terminal_states=None):
        """
        Parameters
        ----------
        actions: list
          The set of actions available.
        states: list
          The state space of the MDP.
        T: ndarray
          An |actions| x |states| x |states| matrix. Entry (a, i, j) gives
          the probability of moving from state i to state j given that action
          a taken.
        R: ndarray
          An |actions| x |states| x |states| matrix. Entry (a, i, j) gives the
          reward for transitioning from state i to state j using action a.
        gamma: float
          Discount factor.
        initial_state: State
          A state in `states` which the MDP will be reset
          to whenever the `reset` method is called.
        terminal_states: list
          A (possibly empty) subset of the states in states. Episodes are
          terminated when the agent enters one of these states.
        """

        self.actions = actions
        self.states = states

        self.initial_state = initial_state
        self.terminal_states = terminal_states

        self._T = T.copy()
        self._T.flags.writeable = False

        assert all(np.sum(self._T, axis=2))
        assert all(self._T > 0) and all(self._T < 1)
        assert self._T.shape == (len(actions), len(states), len(states))

        self._R = R.copy()
        self._R.flags.writeable = False
        assert self._R.shape == (len(actions), len(states), len(states))

        self.gamma = gamma

        self.reset()

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

    def execute_action(self, action=None):
        """
        Play the given action.

        Returns the next state and the reward.
        """

        if action is None:
            if self.n_actions > 1:
                raise ValueError(
                    "Must supply a valid action to "
                    "execute_action when n_actions > 1")
            else:
                action = 0

        if not isinstance(action, Action) and not isinstance(action, int):
            raise ValueError(
                "Action must either be an integer or an instance of the "
                "Action class. Got object of type %s instead." % type(action))

        if isinstance(action, int):
            action = self.actions[action]

        prev_state = self.current_state
        sample = np.random.multinomial(1, self._T[action, self.current_state])
        self.current_state = self.states[np.where(sample > 0)[0][0]]

        reward = self.get_reward(action, prev_state, self.current_state)

        return self.current_state, reward

    @property
    def n_actions(self):
        return len(self.actions)

    @property
    def n_states(self):
        return len(self.states)

    def in_terminal_state(self):
        return (
            self.has_terminal_states()
            and self.current_state in self.terminal_states)

    def has_terminal_states(self):
        return (
            self.terminal_states is not []
            and self.terminal_states is not None)

    @property
    def T(self):
        return self._T

    @property
    def R(self):
        return self._R

    def get_expected_reward(self, a, s):
        return self.T[a, s, :].dot(self.R[a, s, :])

    def get_reward(self, a, s, s_prime=None):
        if s_prime is None:
            return self.get_expected_reward(a, s)
        else:
            return self.R[a, s, s_prime]

    def is_continuous(self):
        return False

    def sample_trajectory(
            self, policy=None, horizon=None, reset=True,
            init=None, return_reward=True, display=False):
        """
        If horizon is None, then trajectory will continue until the episode
        ends (i.e. a terminal state is reached).

        If the mdp has only one action, then policy's 'get_action' method will
        not be called.

        display is the number of seconds to wait between steps.

        Not implemented yet:
        If policy is a tuple, then the first policy will be treated as an
        exploration policy, and the second will be treated as an estimation
        policy (i.e. a policy we are trying to learn about).
        """

        if horizon is None and not self.has_terminal_states():
            raise ValueError(
                "Must supply a finite horizon to sample trajectory from an "
                "MDP that lacks terminal states.")

        if reset:
            if isinstance(init, np.ndarray) or isinstance(init, list):
                sample = np.random.multinomial(1, init)
                init = self.states[np.where(sample > 0)[0][0]]

            self.reset(init)

        if display:
            display = float(display)
            assert display > 0.0

        if not policy:
            policy = UniformRandomPolicy(self)

        if self.n_actions != 1 and not hasattr(policy, 'get_action'):
            raise ValueError(
                "Must supply a policy with a get_action method to sample "
                "a trajectory on an MDP with multiple actions.")

        policy.reset(self.current_state)

        trajectory = []

        if display:
            print "*" * 80

        terminated = False
        i = 0

        while not terminated:
            if display:
                print str(self)

            s = self.current_state

            if self.n_actions == 1:
                a = 0
                s_prime, r = self.execute_action(a)
            else:
                a = policy.get_action()
                s_prime, r = self.execute_action(a)

            if return_reward:
                trajectory.append((s, a, r))
            else:
                trajectory.append((s, a))

            policy.update(a, s_prime, r)

            i += 1

            if display:
                print "Action:", a
                print "New state:", s_prime
                print "Reward:", r
                time.sleep(display)

            terminated = (
                self.in_terminal_state()
                or horizon is not None and i >= horizon)

        a = 0 if self.n_actions == 1 else policy.get_action()
        policy.update(a, self.current_state)

        if display:
            print str(self)

        return trajectory


class SingleActionMDP(MDP):
    """
    Create a single-action mdp (aka a markov chain) by merging
    a multi-action MDP and a policy.
    """

    def __init__(self, mdp, policy):
        self.mdp = mdp
        self.policy = policy
        self.actions = [0]
        self.gamma = mdp.gamma
        self.states = self.mdp.states

    @property
    def name(self):
        return "SingleActionMDP: " + self.mdp.name

    def __str__(self):
        return str(self.mdp)

    def reset(self, state=None):
        self.mdp.reset(state)
        self.policy.reset(self.mdp.current_state)

    def execute_action(self, action=None):
        """ Ignores the given action, uses the action from the policy. """
        a = self.policy.get_action()
        s_prime, r = self.mdp.execute_action(a)
        self.policy.update(a, s_prime, r)
        return s_prime, r

    def in_terminal_state(self):
        return self.mdp.in_terminal_state()

    def has_terminal_states(self):
        return self.mdp.has_terminal_states()

    # TODO: T and R are not really correct, should take policy into account.
    @property
    def T(self):
        return self.mdp.T

    @property
    def R(self):
        return self.mdp.R

    @property
    def current_state(self):
        return self.mdp.current_state


class LinearRewardMDP(MDP):
    def __init__(self, mdp, feature_extractor, theta):
        self.mdp = mdp
        self.feature_extractor = feature_extractor
        self.theta = theta
        self.gamma = self.mdp.gamma

        self.make_R()

    @property
    def name(self):
        return "LinearRewardMDP"

    def __str__(self):
        return str(self.mdp)

    @property
    def states(self):
        return self.mdp.states

    @property
    def actions(self):
        return self.mdp.actions

    def reset(self, state=None):
        self.mdp.reset(state)

    def execute_action(self, action=None):
        if action is None:
            if self.n_actions > 1:
                raise ValueError(
                    "Must supply a valid action to "
                    "execute_action when n_actions > 1")
            else:
                action = 0

        if not isinstance(action, Action) and not isinstance(action, int):
            raise ValueError(
                "Action must either be an integer or an instance of the "
                "Action class. Got object of type %s instead." % type(action))

        if isinstance(action, int):
            action = self.actions[action]

        prev_state = self.mdp.current_state
        s, _ = self.mdp.execute_action(action)
        r = self.get_reward(action, prev_state)

        return s, r

    @property
    def n_actions(self):
        return self.mdp.n_actions

    @property
    def n_states(self):
        return self.mdp.n_states

    def in_terminal_state(self):
        return self.mdp.in_terminal_state()

    def has_terminal_states(self):
        return self.mdp.has_terminal_states()

    @property
    def T(self):
        return self.mdp.T

    @property
    def R(self):
        if self._R is None:
            raise ValueError(
                "Cannot get reward operator of environment "
                "with continuous state space.")
        else:
            return self._R

    def get_reward(self, a, s, s_prime=None):
        state_features = self.feature_extractor.as_vector(s, a)
        return self.theta.dot(state_features)

    @property
    def current_state(self):
        return self.mdp.current_state

    def make_R(self):
        if self.mdp.is_continuous():
            self._R = None
        else:
            R = np.zeros((self.n_actions, self.n_states, self.n_states))

            for a in self.actions:
                for s in self.states:
                    R[a, s, :] = self.get_reward(a, s)

            self._R = R
            self._R.flags.writeable = False


def evaluate_policy(mdp, policy=None, threshold=0.00001):

    if policy is None and mdp.n_actions != 1:
        raise ValueError(
            "Must supply a policy to find value function of an MDP with "
            "multiple actions.")

    j = 0

    V = np.zeros(mdp.n_states)
    old_V = np.inf * np.ones(V.shape)

    T = mdp.T
    R = mdp.R
    gamma = mdp.gamma

    while np.linalg.norm(V - old_V, ord=np.inf) > threshold:
        old_V[:] = V

        for s in mdp.states:

            if mdp.n_actions == 1:
                a = 0
            else:
                policy.reset(s)
                a = policy.get_action()

            V[s] = np.dot(T[a, s, :], R[a, s, :] + gamma * V)

        j += 1

    print "Value function converged after {0} iterations".format(j)

    return V
