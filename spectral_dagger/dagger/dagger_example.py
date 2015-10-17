from spectral_dagger.mdp import MDP, Action, MDPPolicy
from spectral_dagger.envs import grid_world

import numpy as np

from sklearn.linear_model import LogisticRegression


class BadExampleMDP(MDP):
    """
    Dagger will fail on this environment when learning a linear classfier.
    """

    actions = [Action(0, 'red'), Action(1, 'blue')]

    def __init__(self, reset_prob=0.0):
        assert reset_prob >= 0 and reset_prob <= 1

        A_0 = BadExampleMDP.actions[0]
        A_1 = BadExampleMDP.actions[1]

        self.actions = [A_0, A_1]
        self.gamma = 1.0

        self.state_positions = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [1.0, -1.0],
            [2.0, -2.0]
        ])

        dimension = len(self.state_positions)
        self.states = [
            grid_world.GridState(p, i, dimension)
            for i, p in enumerate(self.state_positions)]

        self.initial_state = self.states[0]

        self.observations = self.states

        n_states = len(self.states)
        self._T = np.zeros((2, n_states, n_states))

        self._T[A_0][range(5), [1, 2, 1, 4, 3]] = 1.0 - reset_prob
        self._T[A_0][range(5), 0] = reset_prob

        self._T[A_1][range(5), [3, 2, 1, 4, 3]] = 1.0 - reset_prob
        self._T[A_1][range(5), 0] = reset_prob

        self._R = np.zeros((2, n_states, n_states))

        self.reset()

    @property
    def name(self):
        return "BadExampleMDP"

    def execute_action(self, action):
        o, r = super(BadExampleMDP, self).execute_action(action)
        self.current_state = self.states[self.current_state]
        return self.current_state, r

    def reset(self, init_dist=None):
        self.current_state = self.initial_state

    def generate_observation(self):
        return self.current_state


class GoodExampleMDP(MDP):
    """
    Dagger will succeed on this environment when learning a linear classfier.
    """

    actions = [Action(0, 'red'), Action(1, 'blue')]

    def __init__(self, reset_prob=0.0):
        assert reset_prob >= 0 and reset_prob <= 1

        A_0 = GoodExampleMDP.actions[0]
        A_1 = GoodExampleMDP.actions[1]

        self.actions = [A_0, A_1]

        self.state_positions = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [1.0, -2.0],
            [2.0, -1.0],
            [3.0, -3.0],
            [2.0, -5.0],
            [1.0, -4.0],
        ])

        dimension = len(self.state_positions)
        self.states = [
            grid_world.GridState(p, i, dimension)
            for i, p in enumerate(self.state_positions)]

        self.initial_state = self.states[0]

        self.observations = self.states

        n_states = len(self.states)
        self._T = np.zeros((2, n_states, n_states))

        self._T[A_0][range(8), [1, 2, 1, 4, 5, 6, 7, 3]] = 1 - reset_prob
        self._T[A_0][range(8), 0] = reset_prob

        self._T[A_1][range(8), [3, 2, 1, 4, 5, 6, 7, 3]] = 1 - reset_prob
        self._T[A_1][range(8), 0] = reset_prob

        self._R = np.zeros((2, n_states, n_states))

        self.reset()

    @property
    def name(self):
        return "GoodExampleMDP"

    def execute_action(self, action):
        o, r = super(GoodExampleMDP, self).execute_action(action)
        self.current_state = self.states[self.current_state]
        return self.current_state, r

    def reset(self, init_dist=None):
        self.current_state = self.initial_state

    def generate_observation(self):
        return self.current_state


if __name__ == "__main__":

    import dagger
    from dagger import StateClassifier
    from utils import make_beta

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("type", type=str)
    args = parser.parse_args()

    if args.type == "good":
        env = GoodExampleMDP(0.1)
        good_mapping = {0: 1, 1: 0, 2: 1, 3: 1, 4: 0, 5: 1, 6: 0, 7: 1}
        good_mapping = {
            s: GoodExampleMDP.actions[a] for s, a in good_mapping.iteritems()}
        expert = MDPPolicy(good_mapping)
    elif args.type == "bad":
        env = BadExampleMDP(0.1)
        bad_mapping = {0: 1, 1: 0, 2: 1, 3: 0, 4: 1}
        bad_mapping = {
            s: BadExampleMDP.actions[a] for s, a in bad_mapping.iteritems()}
        expert = MDPPolicy(bad_mapping)
    else:
        raise NotImplementedError(
            "Received invalid type: %s. "
            "Possibilities are 'bad' and 'good'." % args.type)

    learning_algorithm = StateClassifier(LogisticRegression)

    alpha = 0.1
    beta = make_beta(alpha)

    n_iterations = 20
    n_samples = 20
    horizon = 20

    policies = dagger.dagger(
        env, expert, learning_algorithm, beta,
        n_iterations, n_samples, horizon)

    for i, p in enumerate(policies):
        if hasattr(p, 'classifier'):
            print "Policy", i, ": coef: ", p.classifier.coef_
            print "Policy", i, ": intercept: ", p.classifier.intercept_

    n_test_trajectories = 0
    test_horizon = 10

    for i in range(n_test_trajectories):
        trajectory = env.sample_trajectory(
            policies[-1], test_horizon, True, display=1)

    from utils import ABLine2D
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False)

    border = 0.3
    x_pos = [p[0] for p in env.state_positions]
    y_pos = [p[1] for p in env.state_positions]

    xlim = (min(x_pos)-border, max(x_pos)+border)
    ylim = (min(y_pos)-border, max(y_pos)+border)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    x = np.linspace(*xlim, num=20)
    y = np.linspace(*ylim, num=20)
    mesh = map(lambda x: x.flatten(), np.meshgrid(x, y))

    # Plot the states of the mdp
    m = {str(a): a.name for a in env.actions}
    colors = [
        m[policies[-1].classifier.predict(np.array([xp, yp]))[0]]
        for xp, yp in zip(*mesh)]

    plt.scatter(mesh[0], mesh[1], c=colors, alpha=0.3)

    # Plot classification regions for the final policy returned by DAGGER
    exp_mapping = []
    for s in env.states:
        expert.reset(s)
        exp_mapping.append(expert.get_action())

    x, y = zip(*[s.position for s in env.states])
    plt.scatter(x, y, s=np.pi*10**2, c=[a.name for a in exp_mapping])

    # Plot lines representing poliices
    for i, p in enumerate(policies):
        if hasattr(p, 'classifier'):
            ax.add_line(ABLine2D(
                p.classifier.intercept_, p.classifier.coef_[0],
                color=str(0.5 * (1 - float(i)/len(policies)))))

    plt.show()
