"""An implementation of point-based value iteration."""

import numpy as np
from policy import Policy


class PBVI(object):

    def __init__(self):
        pass

    # the m parameter, which essentially controls how ``deep''
    # the search through belief space goes, is quite important
    # when the reward is very delayed
    def fit(self, pomdp, discount, m=6, n=20):
        self.pomdp = pomdp

        V = []

        self.T = pomdp.get_transition_op()
        self.O = pomdp.get_observation_op()

        belief_points = [pomdp.init_dist.copy()]

        self.ops = {}

        for a in pomdp.actions:
            for o in pomdp.observations:
                self.ops[(a, o)] = np.array(
                    [discount * self.T[a, s, :] * self.O[a, :, o]
                     for s in pomdp.states])

        V = np.zeros((pomdp.num_states, len(belief_points)))

        for i in range(m):
            for j in range(n):
                print "Backup : ", j
                V = self.value_backup(V, belief_points)

            if i == m - 1:
                break

            new_belief_points = self.expand_belief_set(
                pomdp, self.T, self.O, belief_points)

            belief_points.extend(new_belief_points)

        self.belief_points = belief_points
        self.V = V

        return BeliefStatePolicy(V, self.T, self.O, pomdp)

    def get_action_value_for_belief(self, gamma_ao, b, a):
        pomdp = self.pomdp

        gamma_ab = np.array(
            [pomdp.get_reward(s, a) for s in pomdp.states])

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
                pomdp.execute_action(a)
                o = pomdp.get_current_observation()

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


class BeliefStatePolicy(Policy):
    def __init__(self, V, T, O, pomdp):
        self.V = V
        self.num_states = pomdp.num_states

        self.T = T
        self.O = O
        self.pomdp = pomdp

        self.b = np.zeros(self.num_states)
        self.b[0] = 1.0

    def reset(self, b=None):
        if b is None:
            b = self.pomdp.init_dist

        self.b = b

    def update(self, action, observation):
        self.last_action = action

        self.b = self.update_belief_state(
            self.b, action, observation)

    def get_value(self, b):
        return max(b.dot(self.V))

    def update_belief_state(self, b, a, o):
        b_prime = b.dot(self.T[a]) * self.O[a, :, o]
        b_prime /= sum(b_prime)

        return b_prime

    def get_action(self):
        b = self.b

        return max(
            self.pomdp.actions,
            key=lambda a: self.get_value(b.dot(self.T[a])))

if __name__ == "__main__":
    import grid_world

    world = np.array([
        ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x'],
        ['x', ' ', ' ', ' ', ' ', ' ', 'x', ' ', ' ', 'x'],
        ['x', ' ', ' ', 'x', 'G', 'x', 'x', ' ', ' ', 'x'],
        ['x', ' ', ' ', 'x', ' ', ' ', ' ', ' ', ' ', 'x'],
        ['x', ' ', ' ', 'x', ' ', 'x', 'x', 'x', ' ', 'x'],
        ['x', 'A', 'x', ' ', ' ', ' ', 'x', ' ', ' ', 'x'],
        ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x']]
    )

    world = np.array([
        ['x', 'x', 'x', 'x', 'x'],
        ['x', ' ', ' ', ' ', 'x'],
        ['x', ' ', ' ', ' ', 'x'],
        ['x', 'x', 'x', ' ', 'x'],
        ['x', 'G', 'x', ' ', 'x'],
        ['x', ' ', ' ', ' ', 'x'],
        ['x', ' ', ' ', ' ', 'x'],
        ['x', 'x', 'x', 'x', 'x']]
    )

    horizon = 20
    discount = 0.99

    num_trials = 3

    means = []
    data = []

    num_trajectories = 10
    colors = [1, 2]

    for num_colors in colors:
        r = []

        for trial in range(num_trials):
            pomdp = grid_world.ColoredGridWorld(num_colors, world)

            print "Training model..."
            pbvi = PBVI()
            policy = pbvi.fit(pomdp, discount)

            trajectory, reward = pomdp.sample_trajectory(
                policy, horizon, True, display=1)

            r.append(sum(reward))

            print "Reward using %d colors: %f" % (num_colors, r[-1])

        data.append(r)
        means.append(np.mean(r))

    import matplotlib.pyplot as plt

    plt.plot(range(len(colors)), means)
    plt.errorbar(range(len(colors)), means, yerr=[np.std(d) for d in data])
    plt.xlim(-1, len(colors))

    plt.show()
