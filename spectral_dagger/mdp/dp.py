import numpy as np

from spectral_dagger import LearningAlgorithm
from spectral_dagger.mdp import GreedyPolicy, MDPPolicy


class PolicyIteration(LearningAlgorithm):

    def __init__(self, threshold=0.0001):
        self.threshold = threshold

    def fit(self, mdp, policy=None):
        if policy is None:
            pi = {}
            for s in mdp.states:
                pi[s] = mdp.actions[self.rng.randint(mdp.n_actions)]

            policy = MDPPolicy(mdp, pi)

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
                    T[a, s, :].dot(R[a, s, :] + gamma * old_V)
                    for a in actions)

            iteration_count += 1

        print "Num iterations for value iteration: ", iteration_count

        self.V = V

        self.actions = actions
        self.states = states
        self.T = T
        self.R = R
        self.gamma = gamma

        return GreedyPolicy(mdp, V)
