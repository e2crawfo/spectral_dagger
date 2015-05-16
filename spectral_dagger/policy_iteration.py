import numpy as np

from mdp import GreedyPolicy, MDPPolicy
from learning_algorithm import LearningAlgorithm


class PolicyIteration(LearningAlgorithm):

    def __init__(self, threshold=0.0001):
        self.threshold = threshold

    def fit(self, mdp, policy=None):
        if policy is None:
            pi = {}
            for s in mdp.states:
                pi[s] = mdp.actions[np.random.randint(mdp.n_actions)]

            policy = MDPPolicy(pi)

        V = np.zeros(mdp.n_states)

        T = mdp.T
        R = mdp.R
        gamma = mdp.gamma

        i = 0

        stable = False
        while not stable:
            old_V = np.inf * np.ones(V.shape)

            print "Starting iteration {0} of policy iteration".format(i)

            j = 0
            # Evaluate policy, obtaining value function
            while np.linalg.norm(V - old_V, ord=np.inf) > self.threshold:
                old_V[:] = V

                for s in mdp.states:
                    policy.reset(s)
                    a = policy.get_action()
                    V[s] = np.dot(T[a, s, :], R[a, s, :] + gamma * V)

                j += 1

            print "Value function converged after {0} iterations".format(j)

            greedy = GreedyPolicy(mdp, V)

            for s in mdp.states:
                policy.reset(s)
                a = policy.get_action()

                greedy.reset(s)
                greedy_a = greedy.get_action()

                stable = a == greedy_a

                if not stable:
                    break

            policy = greedy

            i += 1

        return policy
