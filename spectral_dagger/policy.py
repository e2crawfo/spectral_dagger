import numpy as np


class MDPPolicy(object):

    def __init__(self):
        self.actions = None
        self.state = None

    def fit(self, mdp):
        self.actions = mdp.actions

    def reset(self, state):
        self.state = state

    def update(self, action, state, reward=None):
        self.state = state

    def get_action(self):
        return np.random.choice(self.actions)


class POMDPPolicy(object):

    def __init__(self):
        self.actions = None

    def fit(self, pomdp):
        self.actions = pomdp.actions

    def reset(self, init_dist=None):
        pass

    def update(self, action, observation, reward=None):
        pass

    def get_action(self):
        return np.random.choice(self.actions)
