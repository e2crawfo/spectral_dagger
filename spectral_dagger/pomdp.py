import numpy as np


class State(object):
    def __init__(self, id):
        self.id = id

    def get_id(self):
        return self.id

    def __str__(self):
        return "<State id: %d>" % self.get_id()

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        """Override the default == behavior"""
        if isinstance(other, self.__class__):
            return self.get_id() == other.get_id()
        elif isinstance(other, int):
            return self.get_id() == other

        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, self.__class__) or isinstance(other, int):
            return not self.__eq__(other)

        return NotImplemented

    def __hash__(self):
        """Override default behaviour when used as key in dict"""
        return hash(self.get_id())

    def __index__(self):
        return self.get_id()

    def __int__(self):
        return self.get_id()


class Action(object):
    def __init__(self, id):
        self.id = id

    def get_id(self):
        return self.id

    def __str__(self):
        return "<Action id: %d>" % self.get_id()

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        """Override the default == behavior"""
        if isinstance(other, self.__class__):
            return self.get_id() == other.get_id()
        elif isinstance(other, int):
            return self.get_id() == other

        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, self.__class__) or isinstance(other, int):
            return not self.__eq__(other)

        return NotImplemented

    def __hash__(self):
        """Override default behaviour when used as key in dict"""
        return hash(self.get_id())

    def __index__(self):
        return self.get_id()

    def __int__(self):
        return self.get_id()


class Observation(object):
    def __init__(self, id):
        self.id = id

    def get_id(self):
        return self.id

    def __str__(self):
        return "<Observation id: %d>" % self.get_id()

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        """Override the default == behavior"""
        if isinstance(other, self.__class__):
            return self.get_id() == other.get_id()
        elif isinstance(other, int):
            return self.get_id() == other

        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, self.__class__) or isinstance(other, int):
            return not self.__eq__(other)

        return NotImplemented

    def __hash__(self):
        """Override default behaviour when used as key in dict"""
        return hash(self.get_id())

    def __index__(self):
        return self.get_id()

    def __int__(self):
        return self.get_id()


class POMDP(object):

    def __init__(
            self, actions, observations, states,
            T, O, init_dist, reward, gamme):
        """
        Parameters
        ----------
        actions: list
          The set of actions available.
        observations: list
          The set of observations available.
        states: list
          The state space of the POMDP.
        T: ndarray
          An |actions| x |states| x |states|  matrix. Entry (a, i, j) gives
          the probability of moving from state i to state j given that action
          a taken.
        O: ndarray
          An |actions| x |states| x |observations| matrix. Entry (a, i, j)
          gives the probability of emitting observation j given that the POMDP
          is in state i and the last action was a.
        init_dist: ndarray
          An |states| array. Entry i gives the probability of starting in
          the ith states.
        reward: ndarray
          An |actions| x |states| matrix. Entry (a, i) gives the reward for
          ending up in state i given we took action a.
        """

        self.actions = actions
        self.states = states
        self.observations = observations

        self.T = T
        self.O = O
        self.init_dist = init_dist
        self.reward = reward

        self.current_state = init_dist.sample()

    @property
    def name(self):
        return "DefaultPOMDP"

    def reset(self, init_dist=None):
        """Reset the state using a sample from the init distribution."""
        if init_dist is None:
            init_dist = self.init_dist

        sample = np.random.multinomial(1, init_dist)
        self.current_state = np.where(sample > 0)[0][0]

    @property
    def num_actions(self):
        return len(self.actions)

    @property
    def num_observations(self):
        return len(self.observations)

    @property
    def num_states(self):
        return len(self.states)

    def execute_action(self, action):
        """
        Update the POMDP given that action action has been played.
        """

        if isinstance(action, Action):
            action = action.get_id()
        elif not isinstance(action, int):
            raise ValueError(
                "Action must either be an integer or an instance of the "
                "Action class. Got object of type %s instead." % type(action))

        sample = np.random.multinomial(1, self.T[action, self.current_state])
        self.current_state = np.where(sample > 0)[0][0]

        self.last_action = action

    def get_reward(self, state, action):
        return self.reward[action, state]

    def get_current_reward(self):
        return self.reward[self.last_action, self.current_state]

    def get_current_observation(self):
        sample = np.random.multinomial(
            1, self.O[self.last_action, self.current_state])
        observation = np.where(sample > 0)[0][0]

        return Observation(observation)

    def __str__(self):
        return "%s. Current state: %s" % (
            self.get_name(), str(self.current_state))

    def sample_trajectory(
            self, policy, horizon, reset=True, init_dist=None, display=False):

        import time

        if reset:
            self.reset(init_dist)
            policy.reset(init_dist)

        trajectory = []
        reward = []

        if display:
            print "*" * 80

        for i in range(horizon):
            if display:
                print str(self)

            action = policy.get_action()
            self.execute_action(action)
            policy.action_played(action)

            obs = self.get_current_observation()
            policy.observation_emitted(obs)
            reward.append(self.get_current_reward())

            if display:
                print action
                print obs
                time.sleep(0.3)

            trajectory.append((action, obs))

        if display:
            print str(self)

        return trajectory, reward

    def get_transition_ops(self):
        return self.T

    def get_observation_ops(self):
        return self.O
