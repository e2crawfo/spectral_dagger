import numpy as np


class Policy(object):
    def __init__(self, actions, observations):
         pass

    def reset(self, init_dist=None):
        pass

    def update(self, action, observation):
        pass

    def get_action(self):
        """
        Returns the action chosen by the policy, given the history of
        actions and observations it has encountered. Note that it should Note
        assume that the action that it returns actually gets played.
        """

        pass


class RandomPolicy(Policy):
    def __init__(self, actions, observations):
        self.actions = actions
        self.observations = observations

    def get_action(self):
        return np.random.choice(self.actions)
