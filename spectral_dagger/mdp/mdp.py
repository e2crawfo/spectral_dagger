import numpy as np

from spectral_dagger import Environment, Space
from spectral_dagger.utils import sample_multinomial


class MDP(Environment):

    def __init__(
            self, actions, states, T, R, gamma,
            init_dist=None, terminal_states=None):
        """ A Markov Decision Process.

        Parameters
        ----------
        actions: list
            The list of actions.
        states: list
            The list of states.
        T: ndarray
            An |actions| x |states| x |states| matrix. Entry (a, i, j) gives
            the probability of moving from state i to state j given that action
            a is taken.
        R: ndarray
            An |actions| x |states| x |states| matrix. Entry (a, i, j) gives
            the reward for transitioning from state i to state j with action a.
        gamma: float
            Discount factor.
        init_dist: A distribution over states.
            A distribution from which an initial state will be chosen whenever
            the `reset` method is called.
        terminal_states: list
            A (possibly empty) subset of the states in states. Episodes are
            terminated when the agent enters one of these states.

        """
        self.actions = actions
        self.states = states

        self.init_dist = init_dist
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

    def __str__(self):
        return "%s. Current state: %s" % (
            self.__class__.__name__, str(self.current_state))

    @property
    def n_actions(self):
        return len(self.actions)

    @property
    def n_states(self):
        return len(self.states)

    @property
    def action_space(self):
        return Space([set(self.actions)], "ActionSpace")

    @property
    def observation_space(self):
        return Space([set(self.states)], "ObsSpace")

    def has_reward(self):
        return True

    def has_terminal_states(self):
        return (
            hasattr(self, 'terminal_states') and
            bool(self.terminal_states))

    def in_terminal_state(self):
        return (
            self.has_terminal_states() and
            self.current_state in self.terminal_states)

    def reset(self, init_dist=None):
        """ Resets the state of the MDP.

        Parameters
        ----------
        init_dist: array-like (optional)
            A distribution to choose the initial state from.

        """
        init_dist = self.init_dist if init_dist is None else init_dist
        init_state = self.states[sample_multinomial(init_dist, self.rng)]
        self.current_state = init_state
        return init_state

    def step(self, action=None):
        """ Execute the given action. Returns new state and reward. """

        action = self.action_space.validate(action)
        prev_state = self.current_state

        dist = self._T[action, self.current_state]
        self.current_state = self.states[sample_multinomial(dist, self.rng)]

        reward = self.get_reward(action, prev_state, self.current_state)

        return self.current_state, reward

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


class SingleActionMDP(MDP):
    """ A Markov Chain created by binding an MDP to a policy. """

    def __init__(self, mdp, policy):
        self.mdp = mdp
        self.policy = policy
        self.actions = [0]
        self.gamma = mdp.gamma
        self.states = self.mdp.states

    def __str__(self):
        return str(self.mdp)

    def reset(self, state=None):
        self.mdp.reset(state)
        self.policy.reset(self.mdp.current_state)
        return self.mdp.current_state

    def step(self, action=None):
        """ Ignores the given action, uses the action from the policy. """
        a = self.policy.get_action()
        s_prime, r = self.mdp.step(a)
        self.policy.update(s_prime, a, r)
        return s_prime, r

    def in_terminal_state(self):
        return self.mdp.in_terminal_state()

    def has_terminal_states(self):
        return self.mdp.has_terminal_states()

    # TODO: T and R are not really correct, need to take policy into account.
    @property
    def T(self):
        return self.mdp.T

    @property
    def R(self):
        return self.mdp.R

    @property
    def current_state(self):
        return self.mdp.current_state


class AlternateRewardMDP(MDP):
    """ An MDP created from another MDP, using an alternate reward. """

    def __init__(self, mdp, reward_func):
        self.mdp = mdp
        self.get_reward = reward_func
        self.gamma = self.mdp.gamma

        self.make_R()

    def __str__(self):
        return str(self.mdp)

    @property
    def actions(self):
        return self.mdp.actions

    @property
    def states(self):
        return self.mdp.states

    def reset(self, state=None):
        self.mdp.reset(state)
        return self.mdp.current_state

    def step(self, action=None):
        action = self.action_space.validate(action)

        prev_state = self.current_state
        s, _ = self.mdp.step(action)
        r = self.get_reward(action, prev_state, s)

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


class TimeDependentRewardMDP(AlternateRewardMDP):
    def __init__(self, mdp, reward_func):
        self.mdp = mdp
        self.get_reward = reward_func
        self.gamma = self.mdp.gamma
        self.t = 0

    def reset(self, state=None):
        self.mdp.reset(state)
        self.t = 0
        return self.mdp.current_state

    def step(self, action=None):
        prev_state = self.current_state
        s, _ = self.mdp.step(action)
        r = self.get_reward(action, prev_state, s, self.t)

        self.t += 1

        return s, r

    @property
    def R(self):
        raise RuntimeError(
            "Reward operator does not exist for instances of "
            "TimeDependentReward since the reward is time-dependent.")


class LinearRewardMDP(AlternateRewardMDP):
    def __init__(self, mdp, feature_extractor, theta):
        self.mdp = mdp
        self.feature_extractor = feature_extractor
        self.theta = theta
        self.gamma = self.mdp.gamma

        self.make_R()

    def get_reward(self, a, s, s_prime=None):
        state_features = self.feature_extractor.as_vector(s, a)
        return self.theta.dot(state_features)


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
