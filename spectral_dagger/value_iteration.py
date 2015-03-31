import numpy as np
from policy import MDPPolicy


class ValueIterationPolicy(MDPPolicy):

    def __init__(self, threshold=0.0001):
        self.threshold = threshold

    def fit(self, mdp, V_0=None):

        if V_0 is None:
            V_0 = np.ones(mdp.num_states)

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
                T_s = T[:, s, :]
                V[s] = max(
                    T_s[a, :].dot(R[a, :] + gamma * old_V) for a in actions)

            iteration_count += 1

        print "Num iterations for value iteration: ", iteration_count

        self.V = V

        self.actions = actions
        self.states = states
        self.T = T
        self.R = R
        self.gamma = gamma

    def reset(self, state):
        self.current_state = state

    def update(self, action, next_state, reward):
        self.current_state = next_state

    def get_action(self):
        T_s = self.T[:, self.current_state, :]

        return max(
            self.actions,
            key=lambda a: T_s[a, :].dot(self.R[a, :] + self.gamma * self.V))


def test_value_iteration():
    import grid_world

    world = np.array([
        ['x', 'x', 'x', 'x', 'x', 'x', 'x'],
        ['x', ' ', 'x', ' ', ' ', ' ', 'x'],
        ['x', 'G', 'x', ' ', ' ', ' ', 'x'],
        ['x', ' ', 'x', ' ', ' ', ' ', 'x'],
        ['x', ' ', ' ', ' ', ' ', ' ', 'x'],
        ['x', ' ', ' ', 'x', ' ', 'A', 'x'],
        ['x', 'x', 'x', 'x', 'x', 'x', 'x']]
    )

    gamma = 0.9

    env = grid_world.GridWorld(world, gamma)

    threshold = 0.0001

    policy = ValueIterationPolicy(threshold)
    policy.fit(env)

    horizon = 20
    trajectory = env.sample_trajectory(
        policy, horizon, reset=True, display=True)

    print trajectory

if __name__ == "__main__":
    test_value_iteration()
