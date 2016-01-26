""" An implementation of point-based value iteration. """

import numpy as np

from spectral_dagger import LearningAlgorithm
from spectral_dagger.pomdp import BeliefStatePolicy


class PBVI(LearningAlgorithm):

    def __init__(self, m=6, n=20):
        self.m = m
        self.n = n

    def fit(self, pomdp):
        m = self.m
        n = self.n

        self.pomdp = pomdp

        discount = pomdp.gamma

        V = []

        self.T = pomdp.T
        self.O = pomdp.O

        belief_points = [pomdp.init_dist]

        self.ops = {}

        for a in pomdp.actions:
            for o in pomdp.observations:
                self.ops[(a, o)] = np.array(
                    [discount * self.T[a, s, :] * self.O[a, :, o]
                     for s in pomdp.states])

        V = np.zeros((pomdp.n_states, len(belief_points)))

        for i in range(m):
            for j in range(n):
                print "PBVI Backup Iteration", j
                V = self.value_backup(V, belief_points)

            if i == m - 1:
                break

            new_belief_points = self.expand_belief_set(
                pomdp, self.T, self.O, belief_points)

            belief_points.extend(new_belief_points)

        self.belief_points = belief_points
        self.V = V

        return PBVIPolicy(pomdp, V)

    def get_action_value_for_belief(self, gamma_ao, b, a):
        pomdp = self.pomdp

        gamma_ab = np.array(
            [pomdp.get_reward(a, s) for s in pomdp.states])

        for obs in pomdp.observations:
            gamma = gamma_ao[(a, obs)]
            gamma_ab += gamma[np.argmax(gamma.dot(b)), :]

        return gamma_ab.dot(b), gamma_ab, a

    def value_backup(self, V, belief_points):
        pomdp = self.pomdp

        gamma_ao = {}
        for action in pomdp.actions:
            for obs in pomdp.observations:
                gamma_ao[(action, obs)] = self.ops[(action, obs)].dot(V).T

        V = []
        V_actions = []

        for b in belief_points:
            action_values = (
                self.get_action_value_for_belief(gamma_ao, b, a)
                for a in pomdp.actions)

            max_value, max_vector, max_action = max(
                action_values, key=lambda x: x[0])

            V.append(max_vector)
            V_actions.append(max_action)

        V = np.array(V).T

        return V

    def expand_belief_set(self, pomdp, T, O, belief_points):
        new_belief_points = []

        for b in belief_points:
            current_beliefs = []

            for a in pomdp.actions:
                pomdp.reset(b)
                o, r = pomdp.update(a)

                b_prime = b.dot(T[a]) * O[a, :, o]
                b_prime /= sum(b_prime)

                current_beliefs.append(b_prime)

            min_l1 = lambda b1: min(
                np.linalg.norm(b1 - b2, ord=0) for b2 in belief_points)

            new_belief = max(current_beliefs, key=min_l1)

            is_new_belief = all(
                any(new_belief != old_belief)
                for old_belief in belief_points)

            if is_new_belief:
                new_belief_points.append(new_belief)

        return new_belief_points


class PBVIPolicy(BeliefStatePolicy):

    def __init__(self, pomdp, V):
        self.pomdp = pomdp
        self.V = V.copy()

        self.b = None

    def get_belief_state_value(self, b):
        return max(b.dot(self.V))

    def get_action(self):
        if self.b is None:
            raise Exception(
                "PBVIPolicy has not been given an initial "
                "distribution. Use PBVIPolicy.reset")

        b = self.b

        return max(
            self.pomdp.actions,
            key=lambda a: self.get_belief_state_value(b.dot(self.pomdp.T[a])))
