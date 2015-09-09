import numpy as np


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

    def action_distribution(self, s):
        raise NotImplemented(
            "No action distribution defined for this policy.")


class UniformRandomPolicy(MDPPolicy):

    def __init__(self, mdp):
        self.actions = mdp.actions

    def get_action(self):
        return np.random.choice(self.actions)

    def action_distribution(self, s):
        d = np.ones(len(self.actions))
        return d / sum(d)


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

    def reset(self, state):
        self.current_state = state

    def update(self, action, state, reward=None):
        self.current_state = state

    def get_action(self):
        probs = self.action_distribution(self.current_state)
        sample = np.random.multinomial(1, probs)
        action = np.where(sample > 0)[0][0]

        return action

    def action_distribution(self, state):
        feature_vectors = np.array([
            self.feature_extractor.as_vector(state, a)
            for a in self.actions])
        probs = np.exp(self.temperature * feature_vectors.dot(self.theta))
        probs = probs / sum(probs)

        return probs

    def gradient_log(self, s, a):
        feature_vectors = np.array([
            self.feature_extractor.as_vector(s, b)
            for b in self.actions])

        grad_log = (
            self.feature_extractor.as_vector(s, a)
            - feature_vectors.T.dot(self.action_distribution(s)))

        return grad_log

    def gradient(self, s, a):
        """
        Uses the identity:
            grad(pi(a | s)) = pi(a | s) * grad(log pi(a | s))
        """
        grad = self.action_distribution(s)[a] * self.gradient_log(s, a)
        return grad


class GridKeyboardPolicy(MDPPolicy):
    def __init__(self, mapping=None):
        if mapping is None:
            mapping = {'w': 0, 'd': 1, 's': 2, 'a': 3}

        self.mapping = mapping

    def get_action(self):
        x = raw_input()
        assert len(x) == 1
        return self.mapping[x]
