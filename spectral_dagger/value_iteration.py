import numpy as np

from mdp import GreedyPolicy
from learning_algorithm import LearningAlgorithm


class ValueIteration(LearningAlgorithm):

    def __init__(self, threshold=0.0001):
        self.threshold = threshold

    def fit(self, mdp, V_0=None):

        if V_0 is None:
            V_0 = np.ones(mdp.n_states)

        V = V_0.copy()
        old_V = np.inf * np.ones(V.shape)

        T = mdp.T
        R = mdp.R
        gamma = mdp.gamma
        actions = mdp.actions
        states = mdp.states

        iteration_count = 0
        while np.linalg.norm(V - old_V, ord=np.inf) > self.threshold:
            V, old_V = old_V, V

            for s in states:
                V[s] = max(
                    T[a, s, :].dot(R[a, s, :] + gamma * old_V) for a in actions)

            iteration_count += 1

        print "Num iterations for value iteration: ", iteration_count

        self.V = V

        self.actions = actions
        self.states = states
        self.T = T
        self.R = R
        self.gamma = gamma

        return GreedyPolicy(mdp, V)
