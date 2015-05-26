import numpy as np
from mdp import MDPPolicy


class TD(MDPPolicy):
    """ A prediction policy. Learns the value function of a policy
    acting on an MDP, but does not execute actions."""

    def __init__(self, mdp, alpha, L=0, V_0=None):
        self.states = mdp.states
        self.actions = mdp.actions
        self.gamma = mdp.gamma

        self.alpha = alpha
        self.L = L

        if L > 0:
            self.eligibility_trace = np.zeros(mdp.n_states)

        self.active_traces = []

        if V_0 is None:
            V_0 = 100 * np.ones(mdp.n_states)

        self.V = V_0.copy()

    def reset(self, state):
        self.current_state = state

        if self.L > 0:
            self.eligibility_trace[:] = 0
            self.active_traces = []

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
        if hasattr(self._epsilon, 'next'):
            try:
                next_epsilon = self._epsilon.next()
                self.epsilon = next_epsilon
            except StopIteration:
                pass

        if hasattr(self._alpha, 'next'):
            try:
                next_alpha = self._alpha.next()
                self.alpha = next_alpha
            except StopIteration:
                pass

    def get_action(self):
        p = np.random.random(1)[0]

        if p < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return max(
                (a for a in self.actions),
                key=lambda a: self.Q[self.current_state, a])


class QLearning(ControlTD):
    def __init__(
            self, mdp, alpha, epsilon=0.1, Q_0=None):

        self.states = mdp.states
        self.actions = mdp.actions
        self.gamma = mdp.gamma

        self.alpha = alpha

        if Q_0 is None:
            Q_0 = 100 * np.ones((mdp.n_states, mdp.n_actions))
        self.Q = Q_0.copy()

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
            Q = self.Q
            Q[prev_state, action] += self.alpha * (
                reward + self.gamma * max(Q[state, a] for a in self.actions)
                - Q[prev_state, action])

        self.current_state = state


class Sarsa(ControlTD):
    def __init__(
            self, mdp, alpha, L=0, epsilon=0.1, Q_0=None):
        """
        L is the lambda parameter for TD sarsa. L = 0 equivalent to
        vanilla SARSA
        """

        self.states = mdp.states
        self.actions = mdp.actions
        self.gamma = mdp.gamma

        self.L = L

        # list of lists of length 2. 1st item is (s, a) pair,
        # second item is eligibility trace value
        if L > 0:
            self.eligibility_trace = np.zeros((mdp.n_states, mdp.n_actions))

        self.active_traces = []

        if Q_0 is None:
            Q_0 = 1 * np.ones((mdp.n_states, mdp.n_actions))

        self.Q = Q_0.copy()

        self._epsilon = epsilon
        self._alpha = alpha

        if not hasattr(self._epsilon, 'next'):
            self.epsilon = self._epsilon

        if not hasattr(self._alpha, 'next'):
            self.alpha = self._alpha

        self.update_parameters()

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

            Q = self.Q

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

        # self.eligibility_trace *= self.gamma * self.L
