import numpy as np
from mdp import MDPPolicy


class TD(MDPPolicy):
    """ A base class for TD policies."""

    def __init__(self):
        raise NotImplemented("Cannot instantiate abstract base class `TD`.")

    def update_epsilon(self):
        if self.epsilon_schedule:
            try:
                next_epsilon = self.epsilon_schedule.next()
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
                key=lambda a: self.Q[self.current_state, a])


class QLearning(TD):
    def __init__(
            self, mdp, alpha, epsilon=0.1,
            epsilon_schedule=None, Q_0=None):

        self.states = mdp.states
        self.actions = mdp.actions
        self.gamma = mdp.gamma

        self.alpha = alpha

        if Q_0 is None:
            Q_0 = 100 * np.ones((mdp.n_states, mdp.n_actions))
        self.Q = Q_0

        self.epsilon = epsilon
        self.epsilon_schedule = epsilon_schedule
        self.update_epsilon()

    def reset(self, state):
        self.current_state = state
        self.update_epsilon()

    def update(self, action, state, reward=None):
        prev_state = self.current_state

        if reward is not None:
            Q = self.Q
            Q[prev_state, action] += self.alpha * (
                reward + self.gamma * max(Q[state, a] for a in self.actions)
                - Q[prev_state, action])

        self.current_state = state


class Sarsa(TD):
    def __init__(
            self, mdp, alpha, epsilon=0.1,
            epsilon_schedule=None, Q_0=None):

        self.states = mdp.states
        self.actions = mdp.actions
        self.gamma = mdp.gamma

        self.alpha = alpha

        if Q_0 is None:
            Q_0 = 100 * np.ones((mdp.n_states, mdp.n_actions))
        self.Q = Q_0

        self.epsilon = epsilon
        self.epsilon_schedule = epsilon_schedule
        self.update_epsilon()

    def reset(self, state):
        self.prev_state = None
        self.prev_action = None
        self.prev_reward = None

        self.current_state = state

        self.update_epsilon()

    def update(self, action, state, reward=None):

        if self.prev_reward is not None:
            Q = self.Q
            Q[self.prev_state, self.prev_action] += self.alpha * (
                self.prev_reward + self.gamma * Q[self.current_state, action]
                - Q[self.prev_state, self.prev_action])

        self.prev_state = self.current_state
        self.prev_action = action
        self.prev_reward = reward

        self.current_state = state
