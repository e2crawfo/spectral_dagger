import numpy as np

from spectral_dagger import sample_episodes
from spectral_dagger.mdp import MDP


class RandomWalk(MDP):
    """
    p is probability of transitioning right
    """

    def __init__(self, n_states, p=0.5, left_reward=-1.0, right_reward=1.0):
        assert n_states > 2, "RandomWalk needs at least 3 states"

        self.n_states = n_states
        self.p = p
        self.left_reward = left_reward
        self.right_reward = right_reward

        self._T = np.zeros((1, n_states, n_states))

        self._T[0, 0, 0] = 1.0
        self._T[0, -1, -1] = 1.0

        self._T[0, np.arange(1, n_states-1), np.arange(0, n_states-2)] = 1-p
        self._T[0, np.arange(1, n_states-1), np.arange(0, n_states-2)+2] = p
        self._T.flags.writeable = False
        print(self._T)

        self._R = np.zeros((1, n_states, n_states))
        self._R[:, :, 0] = left_reward
        self._R[:, 0, 0] = 0

        self._R[:, :, -1] = right_reward
        self._R[:, -1, -1] = 0

        self._R.flags.writeable = False

        print(self._R)

        self.actions = [0]
        self.states = list(range(n_states))
        self.initial_state = np.floor(n_states / 2.0)
        self.terminal_states = [0, n_states-1]
        self.gamma = 1.0

        self.reset()

    def update(self, action=None):
        """
        Ignore the action
        """

        return super(RandomWalk, self).update(0)

    def __str__(self):
        return ''.join([
            'x' if self.current_state == i else '-'
            for i in range(self.n_states)])


def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


if __name__ == "__main__":

    n_states = 14
    c = RandomWalk(n_states, p=0.5)

    from spectral_dagger.mdp import evaluate_policy, UniformRandomPolicy
    true_Q = evaluate_policy(c, UniformRandomPolicy(c))

    print("True Q: ")
    print(true_Q)

    from spectral_dagger.td import QLearning

    n_iters = 100
    n_episodes = 100
    errors = []
    alphas = np.linspace(0.0, .3, 20)

    for alpha in alphas:
        errors.append([])
        for n in range(n_iters):
            q = QLearning(c, alpha=alpha, Q_0=np.zeros((n_states, 1)))

            sample_episodes(n_episodes, c, q)

            errors[-1].append(rmse(q.Q, true_Q))

    import matplotlib.pyplot as plt
    plt.plot(alphas, [np.mean(e) for e in errors])
    plt.show()
