import numpy as np



def value_iteration(actions, states, T, R, discount, threshold, V_0=None):
    error = np.inf

    if V_0 is None:
        V_0 = np.ones(len(states))

    V = V_0.copy()
    old_V = np.inf * np.ones(V.shape)

    count = 0
    while np.linalg.norm(V - old_V, ord=np.inf) > threshold:
        V, old_V = old_V, V

        for s in states:
            T_s = T[:, s, :]
            V[s] = max(
                T_s[a, :].dot(R[a, :] + discount * old_V) for a in actions)

        count += 1

    print "Num iterations for value iteration: ", count

    return V


class ValueIterationPolicy(object):
    def __init__(
            self, actions, states, initial_state, T, R,
            discount=0.9, threshold=0.0001):

        self.actions = actions
        self.states = states

        self.initial_state = initial_state

        self.T = T
        self.R = R

        self.discount = discount

        self.V = value_iteration(actions, states, T, R, discount, threshold)

    def reset(self, state):
        self.current_state = state

    def update(self, action, state):
        self.current_state = state

    def get_action(self):
        T_s = self.T[:, self.current_state, :]
        return max(
            self.actions,
            key=lambda a: T_s[a, :].dot(self.R[a, :] + self.discount * self.V))

if __name__ == "__main__":

    import grid_world

    world = np.array([
        ['x', 'x', 'x', 'x','x','x','x'],
        ['x', ' ', 'x', ' ',' ',' ','x'],
        ['x', 'G', 'x', ' ','x',' ','x'],
        ['x', ' ', 'x', ' ','x',' ','x'],
        ['x', ' ', ' ', ' ','x',' ','x'],
        ['x', ' ', ' ', 'x',' ','A','x'],
        ['x', 'x', 'x', 'x','x','x','x']]
    )

    env = grid_world.GridWorld(world)

    T = env.get_transition_op()
    R = env.get_reward_op()

    discount = 0.9

    ocp = ValueIterationPolicy(
        env.actions, env.states, env, T, R, discount)

    num_trajectories = 10
    horizon = 20

    for t in range(num_trajectories):
        env.sample_trajectory_as_mdp(
            ocp, horizon, reset=True, display=True)





