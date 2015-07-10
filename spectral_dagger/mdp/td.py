import numpy as np
from mdp import MDPPolicy


class TD(MDPPolicy):
    """ A prediction policy. Learns the value function of a policy
    acting on an MDP, but does not execute actions."""

    def __init__(self, mdp, alpha, L=0, V_0=None):
        self.actions = mdp.actions
        self.gamma = mdp.gamma

        self.L = L

        if L > 0:
            self.eligibility_trace = np.zeros(mdp.n_states)

        self.active_traces = []

        if V_0 is None:
            V_0 = 100 * np.ones((mdp.n_states, mdp.n_actions))
        elif isinstance(V_0, np.ndarray):
            V_0 = V_0.copy()
        else:
            V_0 = np.full((mdp.n_states, mdp.n_actions), V_0)
        self._V = V_0

        self._alpha = alpha

        if not hasattr(self._alpha, 'next'):
            self.alpha = self._alpha

        self.update_parameters()

    def update_parameters(self):
        if hasattr(self._alpha, 'next'):
            try:
                next_alpha = self._alpha.next()
                self.alpha = next_alpha
            except StopIteration:
                pass

    def reset(self, state):
        self.current_state = state

        if self.L > 0:
            self.eligibility_trace[:] = 0
            self.active_traces = []

        self.update_parameters()

    def update(self, action, state, reward=None):
        prev_state = self.current_state

        if self.L > 0:
            self.eligibility_trace[prev_state] += 1
            if prev_state not in self.active_traces:
                self.active_traces.append(prev_state)

        if reward is not None:
            V = self.V
            delta = reward + self.gamma * V[state] - V[prev_state]

            if self.L > 0:
                for s in self.active_traces:
                    et = self.eligibility_trace[s]
                    V[s] += self.alpha * delta * et
            else:
                V[prev_state] += self.alpha * delta

        self.current_state = state

        for s in self.active_traces[:]:
            self.eligibility_trace[s] *= self.gamma * self.L

            if self.eligibility_trace[s] < 0.001:
                self.active_traces.remove(s)

    def get_action(self):
        raise NotImplementedError(
            "Cannot get action from TD policy. Not a control policy.")


class LinearGradientTD(TD):
    """
    feature_extractor: An instance of function_approximation.FeatureExtractor
    """

    def __init__(
            self, mdp, feature_extractor, alpha, L=0, theta_0=None):

        self.gamma = mdp.gamma
        self.L = L

        if theta_0 is not None:
            self.theta = theta_0.flatten()
            assert self.theta.size == feature_extractor.n_features
        else:
            self.theta = np.zeros(feature_extractor.n_features)

        self.eligibility_trace = np.zeros(self.theta.shape)
        self.feature_extractor = feature_extractor

        self._alpha = alpha

        if not hasattr(self._alpha, 'next'):
            self.alpha = self._alpha

        self.update_parameters()

    def reset(self, state):
        self.current_state = state
        self.eligibility_trace[:] = 0
        self.update_parameters()

    def update(self, action, state, reward=None):
        features = self.feature_extractor.as_vector(self.current_state)
        self.eligibility_trace *= self.gamma * self.L
        self.eligibility_trace += features

        prev_state = self.current_state

        if reward is not None:
            delta = reward + self.gamma * self.V(state) - self.V(prev_state)
            self.theta += self.alpha * delta * self.eligibility_trace

        self.current_state = state

    def V(self, state):
        features = self.feature_extractor.as_vector(state)
        return features.dot(self.theta)


class QTD(TD):
    """ A prediction policy. Learns the action-value function (Q(s,a))
    of a policy acting on an MDP, but does not execute actions."""

    def __init__(self, mdp, alpha, L=0, Q_0=None):
        """
        L is the lambda parameter for TD sarsa. L = 0 equivalent to
        vanilla SARSA
        """

        self.actions = mdp.actions
        self.gamma = mdp.gamma

        self.L = L

        if L > 0:
            self.eligibility_trace = np.zeros((mdp.n_states, mdp.n_actions))

        self.active_traces = []

        if Q_0 is None:
            Q_0 = 100 * np.ones((mdp.n_states, mdp.n_actions))
        elif isinstance(Q_0, np.ndarray):
            Q_0 = Q_0.copy()
        else:
            Q_0 = np.full((mdp.n_states, mdp.n_actions), Q_0)
        self._Q = Q_0

        self._alpha = alpha

        if not hasattr(self._alpha, 'next'):
            self.alpha = self._alpha

        self.update_parameters()

        self.prev_reward = None

    def reset(self, state):
        self.prev_state = None
        self.prev_action = None
        self.prev_reward = None

        if self.L > 0:
            self.eligibility_trace[:] = 0
            self.active_traces = []

        self.current_state = state

        self.update_parameters()

    def update(self, action, state, reward=None):

        if self.prev_reward is not None:
            if self.L > 0:
                self.eligibility_trace[self.prev_state, self.prev_action] += 1
                s, a = (self.prev_state, self.prev_action)
                if (s, a) not in self.active_traces:
                    self.active_traces.append((s, a))

            Q = self._Q
            delta = (
                self.prev_reward + self.gamma * Q[self.current_state, action]
                - Q[self.prev_state, self.prev_action])

            if self.L > 0:
                for (s, a) in self.active_traces:
                    et = self.eligibility_trace[s, a]
                    Q[s, a] += self.alpha * delta * et
            else:
                Q[self.prev_state, self.prev_action] += self.alpha * delta

        self.prev_state = self.current_state
        self.prev_action = action
        self.prev_reward = reward

        self.current_state = state

        for s, a in self.active_traces[:]:
            self.eligibility_trace[s, a] *= self.gamma * self.L

            if self.eligibility_trace[s, a] < 0.001:
                self.active_traces.remove((s, a))

    def Q(self, state, action):
        return self._Q[state, action]


class ControlTD(TD):
    """
    A base class for TD policies that include control. Chooses actions
    epsilon-greedily with respect to current estimate of action-value
    function (i.e. Q(s, a)).

    """

    def __init__(self, mc, alpha):
        raise NotImplemented(
            "Cannot instantiate abstract base class `ControlTD`.")

    def update_parameters(self):
        super(ControlTD, self).update_parameters()

        if hasattr(self._epsilon, 'next'):
            try:
                next_epsilon = self._epsilon.next()
                self.epsilon = next_epsilon
            except StopIteration:
                pass

    def get_action(self):
        p = np.random.random(1)[0]

        if p < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return max(
                (a for a in self.actions),
                key=lambda a: self.Q(self.current_state, a))

    def Q(self, state, action):
        return self._Q[state, action]


class QLearning(ControlTD):
    def __init__(
            self, mdp, alpha, epsilon=0.1, Q_0=None):

        self.actions = mdp.actions
        self.gamma = mdp.gamma

        self.alpha = alpha

        if Q_0 is None:
            Q_0 = 100 * np.ones((mdp.n_states, mdp.n_actions))
        elif isinstance(Q_0, np.ndarray):
            Q_0 = Q_0.copy()
        else:
            Q_0 = np.full((mdp.n_states, mdp.n_actions), Q_0)
        self._Q = Q_0

        self._epsilon = epsilon
        self._alpha = alpha

        if not hasattr(self._epsilon, 'next'):
            self.epsilon = self._epsilon

        if not hasattr(self._alpha, 'next'):
            self.alpha = self._alpha

        self.update_parameters()

    def reset(self, state):
        self.current_state = state
        self.update_parameters()

    def update(self, action, state, reward=None):
        prev_state = self.current_state

        if reward is not None:
            Q = self._Q
            Q[prev_state, action] += self.alpha * (
                reward + self.gamma * max(Q[state, a] for a in self.actions)
                - Q[prev_state, action])

        self.current_state = state


class Sarsa(ControlTD):
    def __init__(self, mdp, alpha, L=0, epsilon=0.1, Q_0=None):
        """
        L is the lambda parameter for TD sarsa. L = 0 equivalent to
        vanilla SARSA
        """

        self.actions = mdp.actions
        self.gamma = mdp.gamma

        self.L = L

        if L > 0:
            self.eligibility_trace = np.zeros((mdp.n_states, mdp.n_actions))

        self.active_traces = []

        if Q_0 is None:
            Q_0 = 100 * np.ones((mdp.n_states, mdp.n_actions))
        elif isinstance(Q_0, np.ndarray):
            Q_0 = Q_0.copy()
        else:
            Q_0 = np.full((mdp.n_states, mdp.n_actions), Q_0)
        self._Q = Q_0

        self._epsilon = epsilon
        self._alpha = alpha

        if not hasattr(self._epsilon, 'next'):
            self.epsilon = self._epsilon

        if not hasattr(self._alpha, 'next'):
            self.alpha = self._alpha

        self.update_parameters()

        self.prev_reward = None

    def reset(self, state):
        self.prev_state = None
        self.prev_action = None
        self.prev_reward = None

        if self.L > 0:
            self.eligibility_trace[:] = 0
            self.active_traces = []

        self.current_state = state

        self.update_parameters()

    def update(self, action, state, reward=None):

        if self.prev_reward is not None:
            if self.L > 0:
                self.eligibility_trace[self.prev_state, self.prev_action] += 1
                s, a = (self.prev_state, self.prev_action)
                if (s, a) not in self.active_traces:
                    self.active_traces.append((s, a))

            Q = self._Q
            delta = (
                self.prev_reward + self.gamma * Q[self.current_state, action]
                - Q[self.prev_state, self.prev_action])

            if self.L > 0:
                for (s, a) in self.active_traces:
                    et = self.eligibility_trace[s, a]
                    Q[s, a] += self.alpha * delta * et
            else:
                Q[self.prev_state, self.prev_action] += self.alpha * delta

        self.prev_state = self.current_state
        self.prev_action = action
        self.prev_reward = reward

        self.current_state = state

        for s, a in self.active_traces[:]:
            self.eligibility_trace[s, a] *= self.gamma * self.L

            if self.eligibility_trace[s, a] < 0.001:
                self.active_traces.remove((s, a))


class LinearGradientSarsa(ControlTD):
    def __init__(
            self, mdp, feature_extractor, alpha,
            L=0, epsilon=0.1, theta_0=None):

        """
        We can look at this as using a single parameter vector, theta,
        with length n_features * n_actions. In a given state, each action
        has a different but related representation. Action 0 has the
        state-feature vector in the first n_features locations and is
        0 everywhere else, action 1 has the state-feature vector in the
        second n_features locations and is 0 everywhere else, etc. This is
        the interpretation, but implementationally we store theta in an
        n_features x n_actions matrix. Effectively, we have a separate
        linear value function for each action.
        """
        self.gamma = mdp.gamma
        self.L = L
        self.actions = mdp.actions
        self.n_actions = len(self.actions)

        if theta_0 is not None:
            self.theta = theta_0.flatten()[:]
            self.theta = np.tile(self.theta, (1, self.n_actions))
            assert (
                self.theta.size == feature_extractor.n_features * self.n_actions)
        else:
            self.theta = np.zeros(
                (feature_extractor.n_features, self.n_actions))

        self.eligibility_trace = np.zeros(self.theta.shape)
        self.feature_extractor = feature_extractor

        self._epsilon = epsilon
        self._alpha = alpha

        if not hasattr(self._epsilon, 'next'):
            self.epsilon = self._epsilon

        if not hasattr(self._alpha, 'next'):
            self.alpha = self._alpha

        self.update_parameters()

        self.prev_reward = None

    def reset(self, state):
        self.prev_state = None
        self.prev_action = None
        self.prev_reward = None

        self.eligibility_trace[:] = 0

        self.current_state = state
        self.update_parameters()

    def update(self, action, state, reward=None):
        if self.prev_reward is not None:

            features = self.feature_extractor.as_vector(self.prev_state)
            self.eligibility_trace[:, self.prev_action] += features

            delta = (
                self.prev_reward
                + self.gamma * self.Q(self.current_state, action)
                - self.Q(self.prev_state, self.prev_action))

            self.theta += self.alpha * delta * self.eligibility_trace
            self.eligibility_trace *= self.gamma * self.L

        self.prev_state = self.current_state
        self.prev_action = action
        self.prev_reward = reward

        self.current_state = state

    def Q(self, state, action):
        features = self.feature_extractor.as_vector(state)
        return features.dot(self.theta[:, action])


if __name__ == "__main__":
    from cts_grid_world import ContinuousGridWorld
    from utils import geometric_sequence
    from function_approximation import RectangularTileCoding

    dummy_map = np.array([
        ['x', 'x', 'x', 'x', 'x'],
        ['x', ' ', ' ', 'G', 'x'],
        ['x', ' ', ' ', ' ', 'x'],
        ['x', ' ', ' ', 'P', 'x'],
        ['x', ' ', ' ', 'P', 'x'],
        ['x', ' ', ' ', 'P', 'x'],
        ['x', 'x', 'x', 'x', 'x']])

    dummy_world = ContinuousGridWorld(dummy_map, speed=0.5)
    feature_extractor = RectangularTileCoding(
        n_tilings=5, bounds=dummy_world.world_map.bounds.s, granularity=0.5)
    alpha = geometric_sequence(0.1, 1, 500)
    epsilon = geometric_sequence(0.05, 1, 100)
    L = 0.5

    linear_gtd = LinearGradientSarsa(
        dummy_world, feature_extractor, alpha=alpha, epsilon=epsilon, L=L)

    horizon = 100
    n_episodes = 1000
    n_steps = []
    rewards = []
    for i in range(n_episodes):
        trajectory = dummy_world.sample_trajectory(
            policy=linear_gtd, display=False, reset=True, horizon=horizon)
        print "n_steps: ", len(trajectory)
        print "max theta: ", np.max(np.max(linear_gtd.theta))
        print "min theta: ", np.min(np.min(linear_gtd.theta))
        average_reward = np.mean([t[2] for t in trajectory])
        rewards.append(average_reward)
        n_steps.append(len(trajectory))

    n_episodes = 0
    for i in range(n_episodes):
        trajectory = dummy_world.sample_trajectory(
            policy=linear_gtd, display=True, reset=True, horizon=horizon)
        print "n_steps: ", len(trajectory)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.subplot(3, 1, 1)
    plt.scatter(
        [s[1] for s, a, r in trajectory], [-s[0] for s, a, r in trajectory])
    plt.subplot(3, 1, 2)
    plt.plot(n_steps)
    plt.subplot(3, 1, 3)
    plt.plot(rewards)
    plt.set_xlim((0, ))
    plt.show()
