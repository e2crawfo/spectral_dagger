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
    def __init__(self, mdp, feature_extractor, phi):
        self.mdp = mdp
        self.feature_extractor = feature_extractor
        self.phi = phi.flatten()

        assert len(self.phi == self.feature_extractor.n_features)

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
            for a in self.mdp.actions])
        probs = np.exp(feature_vectors.dot(self.phi))
        probs = probs / sum(probs)

        return probs


class GridKeyboardPolicy(MDPPolicy):
    def __init__(self, mapping=None):
        if mapping is None:
            mapping = {'w': 0, 'd': 1, 's': 2, 'a': 3}

        self.mapping = mapping

    def get_action(self):
        x = raw_input()
        assert len(x) == 1
        return self.mapping[x]
