import numpy as np

from spectral_dagger import Policy, Space
from spectral_dagger.utils import sample_multinomial


class UniformRandomPolicy(Policy):

    def __init__(self, actions):
        self.actions = actions

    @property
    def action_space(self):
        return Space([set(self.actions)], "ActionSpace")

    @property
    def observation_space(self):
        return Space(name="ObsSpace")

    def reset(self, obs=None):
        pass

    def get_action(self):
        return self.run_rng.choice(self.actions)

    def action_distribution(self, s):
        d = np.ones(len(self.actions))
        return d / sum(d)


class MDPPolicy(Policy):
    """ A policy that chooses actions as a function of the previous obs.

    Parameters
    ----------
    mdp: MDP instance
        The MDP that the policy will operate on.
    pi: dict or callable
        A mapping from states to actions.

    """
    def __init__(self, mdp, pi):
        self.is_dict = hasattr(pi, '__getitem__')

        if not self.is_dict and not callable(pi):
            raise Exception(
                "pi must be either a dict or a callable.")

        self.pi = pi

        self.actions = mdp.actions
        self.states = mdp.states

    @property
    def action_space(self):
        return Space(set(self.actions), "ActionSpace")

    @property
    def observation_space(self):
        return Space(set(self.states), "ObsSpace")

    def reset(self, state):
        self.current_state = state

    def update(self, state, action, reward=None):
        self.current_state = state

    def get_action(self):
        if self.is_dict:
            return self.pi[self.current_state]
        else:
            return self.pi(self.current_state)

    def action_distribution(self, s):
        raise NotImplemented("No action distribution defined for this policy.")


class GreedyPolicy(MDPPolicy):

    def __init__(self, mdp, V, epsilon=0.0):
        self.T = mdp.T
        self.R = mdp.R
        self.gamma = mdp.gamma
        self.actions = mdp.actions
        self.states = mdp.states
        self.epsilon = epsilon

        self.V = V.copy()

    def set_value(self, s, v):
        self.V[s] = v

    def get_action(self):
        if self.epsilon > 0 and self.run_rng.rand() < self.epsilon:
            return self.run_rng.choice(self.actions)
        else:
            T_s = self.T[:, self.current_state, :]
            R_s = self.R[:, self.current_state, :]

            return max(
                self.actions,
                key=lambda a: T_s[a, :].dot(R_s[a, :] + self.gamma * self.V))


class LinearGibbsPolicy(MDPPolicy):

    def __init__(self, actions, feature_extractor, theta, temperature=1.0):
        self.actions = actions
        self.feature_extractor = feature_extractor
        self.theta = theta.flatten()
        self.temperature = temperature

        assert len(self.theta == self.feature_extractor.n_features)

    @property
    def action_space(self):
        return Space(set(self.actions), "CtsActionSpace")

    @property
    def observation_space(self):
        return Space(
            [(-np.inf, np.inf)] * self.state_dim, "CtsObsSpace")

    def reset(self, state):
        self.current_state = state

    def update(self, state, action, reward=None):
        self.current_state = state

    def get_action(self):
        probs = self.action_distribution(self.current_state)
        action = sample_multinomial(probs, self.run_rng)

        return action

    def action_distribution(self, state):
        feature_vectors = np.array([
            self.feature_extractor.as_vector(state, a)
            for a in self.actions])
        probs = np.exp(self.temperature * feature_vectors.dot(self.theta))
        probs = probs / sum(probs)

        return probs

    def gradient_log(self, s, a):
        """ Compute the gradient of log(pi(a | s)) wrt theta. """

        feature_vectors = np.array([
            self.feature_extractor.as_vector(s, b)
            for b in self.actions])

        grad_log = (
            self.feature_extractor.as_vector(s, a)
            - feature_vectors.T.dot(self.action_distribution(s)))

        return grad_log

    def gradient(self, s, a):
        """ Compute the gradient of pi(a | s) wrt theta.

        Uses the following identity:
            grad(pi(a | s)) = pi(a | s) * grad(log pi(a | s))
        """
        grad = self.action_distribution(s)[a] * self.gradient_log(s, a)
        return grad
