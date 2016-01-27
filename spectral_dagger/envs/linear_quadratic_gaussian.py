import numpy as np

from spectral_dagger import Space
from spectral_dagger.mdp import MDP, MDPPolicy


class LQG(MDP):
    """ A LinearQuadraticGaussian Environment. """

    def __init__(self, A, B, noise, Q, R):
        self.A = A
        self.B = B
        self.noise = noise
        self.Q = Q
        self.R = R

        self.state_dim = A.shape[0]
        self.action_dim = B.shape[1]

        assert A.shape[1] == self.state_dim
        assert B.shape[0] == self.state_dim
        assert noise.shape[0] == noise.shape[1] == self.state_dim
        assert Q.shape[0] == Q.shape[1] == self.state_dim
        assert R.shape[0] == R.shape[1] == self.action_dim

    @property
    def name(self):
        return "LinearQuadraticGaussian"

    def action_space(self):
        return Space(
            [(-np.inf, np.inf)] * self.action_dim, "CtsActionSpace")

    def observation_space(self):
        return Space(
            [(-np.inf, np.inf)] * self.state_dim, "CtsObsSpace")

    def reset(self, state=None):
        if state is None:
            state = np.zeros(self.state_dim)

        self.x = state

        return state

    def update(self, u):
        prev_state = self.x

        mean = self.A.dot(self.x) + self.B.dot(u)
        self.x = np.random.multivariate_normal(mean, self.noise)
        reward = self.get_reward(u, prev_state, self.x)

        return self.x.copy(), reward

    @property
    def current_state(self):
        return self.x

    def get_reward(self, u, x, x_prime=None):
        return -(x.T.dot(self.Q).dot(x)) - (u.T.dot(self.R).dot(u))


class GaussianController(MDPPolicy):
    """ Parameterized policy of the form N(theta1 x, theta2). """

    def __init__(self, lqg, theta1=None, theta2=None):
        self.lqg = lqg

        if theta1 is None:
            theta1 = np.zeros((lqg.action_dim, lqg.state_dim))
        self.theta1 = theta1

        if theta2 is None:
            theta2 = np.eye(lqg.action_dim)
        self.theta2 = theta2

    def action_space(self):
        return Space(
            [(-np.inf, np.inf)] * self.action_dim, "CtsActionSpace")

    def observation_space(self):
        return Space(
            [(-np.inf, np.inf)] * self.state_dim, "CtsObsSpace")

    def reset(self, x):
        self.x = x

    def update(self, state, action, reward=None):
        pass

    def get_action(self):
        return np.random.multivariate_normal(
            self.theta1.dot(self.x), self.theta2)
