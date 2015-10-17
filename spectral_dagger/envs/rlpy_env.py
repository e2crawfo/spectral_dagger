import numpy as np

from spectral_dagger.mdp import MDP

try:
    from rlpy import Domains
    rlpy_avail = True
except ImportError:
    rlpy_avail = False


class rlpyEnv(MDP):

    def __init__(self, name, *args, **kwargs):
        """ A wrapper for rlpy environments.

        Makes rlpy environments accessible to spectral_dagger.
        python must be able to find rlpy. Currently only works with
        environments that have the same set of actions for all states.

        rlpy domains that are currently supported (possible values for `name`):

        FiniteTrackCartPole

        Parameters
        ----------
        name: string
            Name of the environment class, as an attribute of rlpy.Domains.
        """

        env_class = getattr(Domains, name)
        self.env = env_class(*args, **kwargs)
        self.current_state = None

    @property
    def name(self):
        return "rlpyEnv"

    def __str__(self):

        return str(self.env) + "\nCurrent state: " + str(self.current_state)

    def reset(self, state=None):
        """
        Resets the state of the MDP.

        Parameters
        ----------
        state: anything
            Ignored.

        """
        self.current_state, is_terminal_state, possible_actions = self.env.s0()

    def execute_action(self, action):
        """
        Play the given action.

        Returns the next state and the reward.

        """
        if action is None:
            if self.n_actions > 1:
                raise ValueError(
                    "Must supply a valid action to "
                    "execute_action when n_actions > 1")
            else:
                action = 0
        result = self.env.step(action)
        reward, self.current_state, term_state, poss_actions = result

        return self.current_state, reward

    @property
    def actions(self):
        return self.env.possibleActions()

    @property
    def n_states(self):
        return self.env.states_num

    def has_terminal_states(self):
        raise NotImplementedError()

    def in_terminal_state(self):
        return self.env.isTerminal()

    def is_continuous(self):
        return self.env.states_num == np.inf