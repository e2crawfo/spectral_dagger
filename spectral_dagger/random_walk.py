from spectral_dagger.mdp import MDP

import numpy as np


class RandomWalk(MDP):
    """
    p is probability of transitioning right
    """

    def __init__(
            self, n_states, p=0.5, gamma=0.9,
            left_reward=-1.0, right_reward=1.0):

        assert n_states > 2, "RandomWalk needs at least 3 states"

        self.states = range(n_states)
        self.p = p
        self.gamma = gamma
        self.left_reward = left_reward
        self.right_reward = right_reward

        self._T = np.zeros((1, n_states, n_states))

        self._T[0, 0, 0] = 1.0
        self._T[0, -1, -1] = 1.0

        self._T[0, np.arange(1, n_states-1), np.arange(0, n_states-2)] = 1-p
        self._T[0, np.arange(1, n_states-1), np.arange(0, n_states-2)+2] = p
        self._T.flags.writeable = False

        self._R = np.zeros((1, n_states, n_states))
        self._R[:, :, 0] = left_reward
        self._R[:, 0, 0] = 0

        self._R[:, :, -1] = right_reward
        self._R[:, -1, -1] = 0

        self._R.flags.writeable = False

        self.actions = [0]
        self.states = range(n_states)
        self.initial_state = np.floor(n_states / 2.0)
        self.terminal_states = [0, n_states-1]
        self.gamma = 1.0

        self.reset()

    def execute_action(self, action=None):
        """
        Ignore the action
        """

        return super(RandomWalk, self).execute_action(0)

    def __str__(self):
        return ''.join([
            'x' if self.current_state == i else '-'
            for i in range(self.n_states)])
