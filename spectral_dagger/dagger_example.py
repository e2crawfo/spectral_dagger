from mdp import MDP, Action, MDPPolicy
import grid_world

import numpy as np

from sklearn.linear_model import LogisticRegression


class BadExampleMDP(MDP):
    """
    Dagger will fail on this environment when learning a linear classfier.
    """

    actions = [Action(0, 'red'), Action(1, 'blue')]

    def __init__(self):
        A_0 = BadExampleMDP.actions[0]
        A_1 = BadExampleMDP.actions[1]

        self.actions = [A_0, A_1]

        self.state_positions = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [-1.0, -1.0],
            [-2.0, -2.0]
        ])

        self.states = [
            grid_world.GridState(p, i)
            for i, p in enumerate(self.state_positions)]

        self.initial_state = self.states[0]

        self.observations = self.states

        num_states = len(self.states)
        self._T = np.zeros((2, num_states, num_states))

        self._T[A_0][0][1] = 1.0
        self._T[A_0][1][2] = 1.0
        self._T[A_0][2][1] = 1.0
        self._T[A_0][3][4] = 1.0
        self._T[A_0][4][3] = 1.0

        self._T[A_1][0][3] = 1.0
        self._T[A_1][1][2] = 1.0
        self._T[A_1][2][1] = 1.0
        self._T[A_1][3][4] = 1.0
        self._T[A_1][4][3] = 1.0

        self._R = np.zeros((2, num_states))

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


class BadExampleExpert(MDPPolicy):
    def __init__(self, initial_state):
        self.initial_state = initial_state
        self.current_state = self.initial_state

    def reset(self, init_dist=None):
        self.current_state = self.initial_state

    def update(self, action, state, reward=None):
        self.current_state = state

    def get_action(self):
        mapping = {0: 1, 1: 0, 2: 1, 3: 0, 4: 1}
        return BadExampleMDP.actions[mapping[self.current_state.get_id()]]


class GoodExampleMDP(MDP):
    """
    Dagger will succeed on this environment when learning a linear classfier.
    """

    actions = [Action(0, 'red'), Action(1, 'blue')]

    def __init__(self):
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

        self.states = [
            grid_world.GridState(p, i)
            for i, p in enumerate(self.state_positions)]

        self.initial_state = self.states[0]

        self.observations = self.states

        num_states = len(self.states)
        self.T = np.zeros((2, num_states, num_states))

        self.T[A_0][0][1] = 1.0
        self.T[A_0][1][2] = 1.0
        self.T[A_0][2][1] = 1.0

        self.T[A_0][3][4] = 1.0
        self.T[A_0][4][5] = 1.0
        self.T[A_0][5][6] = 1.0
        self.T[A_0][6][7] = 1.0
        self.T[A_0][7][3] = 1.0

        self.T[A_1][0][3] = 1.0
        self.T[A_1][1][2] = 1.0
        self.T[A_1][2][1] = 1.0

        self.T[A_1][3][4] = 1.0
        self.T[A_1][4][5] = 1.0
        self.T[A_1][5][6] = 1.0
        self.T[A_1][6][7] = 1.0
        self.T[A_1][7][3] = 1.0

        self.reward = np.zeros((2, num_states))

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


class GoodExampleExpert(MDPPolicy):
    def __init__(self, initial_state):
        self.initial_state = initial_state
        self.current_state = self.initial_state

    def reset(self, init_dist=None):
        self.current_state = self.initial_state

    def update(self, action, state, reward=None):
        self.current_state = state

    def get_action(self):
        mapping = {0: 1, 1: 0, 2: 1, 3: 1, 4: 0, 5: 1, 6: 0, 7: 1}
        return GoodExampleMDP.actions[mapping[self.current_state.get_id()]]


class LinearLearner(MDPPolicy):
    def __init__(self, initial_state, actions, observations, data):
        self.initial_state = initial_state
        self.actions = actions
        self.observations = observations

        states = []
        expert_actions = []

        labelled_data = [d for d in data if d[1] is not None]

        for seq, ea_seq in labelled_data:

            state_seq = [initial_state] + [s for a, s in seq]

            for e, s in zip(ea_seq, state_seq):
                states.append(s.position)
                expert_actions.append(str(e))

        states = np.array(states)
        expert_actions = np.array(expert_actions)

        self.classifier = LogisticRegression()
        self.classifier.fit(states, expert_actions)

        self.action_lookup = {str(a): a for a in actions}

        self.reset()

    def reset(self, init_dist=None):
        self.current_state = self.initial_state

    def update(self, action, observation, reward=None):
        self.last_action = action
        self.current_state = observation

    def get_action(self):
        action_string = self.classifier.predict(self.current_state.position)
        return self.action_lookup[action_string[0]]


if __name__ == "__main__":

    import dagger

    if False:
        env = GoodExampleMDP()
        initial_policy = GoodExampleExpert(env.initial_state)
        expert = GoodExampleExpert(env.initial_state)
    else:
        env = BadExampleMDP()
        initial_policy = BadExampleExpert(env.initial_state)
        expert = BadExampleExpert(env.initial_state)

    policy_class = (
        lambda *a, **kw: LinearLearner(env.initial_state, *a, **kw))

    def make_beta(alpha):
        i = 1
        while True:
            yield (1 - alpha)**(i-1)
            i += 1

    alpha = 0.1
    beta = make_beta(alpha)

    num_iterations = 100
    num_samples = 20
    horizon = 20

    policies = dagger.dagger(
        env, initial_policy, policy_class, expert, beta,
        num_iterations, num_samples, horizon)

    for i, p in enumerate(policies):
        if hasattr(p, 'classifier'):
            print "Policy", i, ": coef: ", p.classifier.coef_
            print "Policy", i, ": intercept: ", p.classifier.intercept_

    num_test_trajectories = 10
    test_horizon = 10

    for i in range(num_test_trajectories):
        trajectory = env.sample_trajectory(
            policies[-1], test_horizon, True, display=1)

    # import matplotlib.pyplot as plt
    # from matplotlib.lines import Line2D

    # for i, p in enumerate(policies):
    #     ax = plt.subplot(num_iterations, 1, i)

    #     ax.add_line(Line2D())

    # plt.show()
