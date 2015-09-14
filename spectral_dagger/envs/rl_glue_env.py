import rlglue.RLGlue as rl_glue


class GridWorld(MDP):

    def __init__(
            self, world_map=None, gamma=0.9, noise=0.1,
            rewards=None, terminate_on_goal=True):
        self.gamma = gamma
        self.noise = noise

        if isinstance(world_map, WorldMap):
            self.world_map = world_map
        else:
            self.world_map = WorldMap(world_map)

        self.init_position = self.world_map.init_position
        self.goal_position = self.world_map.goal_position
        self.pit_positions = self.world_map.pit_positions
        self.puddle_positions = self.world_map.puddle_positions
        self.death_positions = self.world_map.death_positions
        self.trap_positions = self.world_map.trap_positions
        self.positions = self.world_map.positions

        self.terminate_on_goal = terminate_on_goal

        self.set_rewards(rewards)

        self.make_T()
        self.make_R()

        self.reset()

    @property
    def name(self):
        return "GridWorld"

    def __str__(self):
        return str(self.world_map)

    def set_rewards(self, rewards):
        if rewards is None:
            rewards = {}
        self.pit_reward = rewards.get('pit', GridWorld.PIT_REWARD)
        self.puddle_reward = rewards.get('puddle', GridWorld.PUDDLE_REWARD)
        self.death_reward = rewards.get('death', GridWorld.DEATH_REWARD)
        self.trap_reward = rewards.get('trap', GridWorld.TRAP_REWARD)
        self.goal_reward = rewards.get('goal', GridWorld.GOAL_REWARD)
        self.default_reward = rewards.get('default', GridWorld.DEFAULT_REWARD)

    def reset(self, state=None):
        """
        Resets the state of the MDP.

        Parameters
        ----------
        state: State or int or or None
          If state is a State or int, sets the current state accordingly.
          If None, states are chosen according to the grid world's initial
          distribution. If self.init_position == None, feasible positions are
          chosen uniformly at random. Otherwise, the initial position is set
          equal to self.init_position.
        """

        if isinstance(state, GridState):
            self.current_position = Position(state.position)
        elif isinstance(state, int):
            self.current_position = self.positions[state]
        elif state is None:
            if self.init_position is None:
                locations = set(self.positions)
                locations -= set(self.pit_positions)
                locations -= set(self.puddle_positions)
                locations -= set(self.death_positions)
                locations -= set(self.trap_positions)
                locations -= set([self.goal_position])

                self.current_position = list(locations)[
                    np.random.randint(len(locations))]
            else:
                self.current_position = Position(self.init_position)

        else:
            try:
                self.current_position = Position(state)
            except:
                raise ValueError(
                    "GridWorld.reset received invalid starting "
                    "state: %s" % state)

    def execute_action(self, action):
        """
        With probability 0.8, move in the specified direction. With
        probability 0.2, move in a randomly chosen perpendicular direction.
        If there is a wall in the direction selected by this random process,
        stay in place.

        Returns the next state and the reward.
        """
        try:
            action = GridAction(action)
        except:
            raise ValueError(
                "Action for GridWorld must either be GridAction, or "
                "something convertible to a GridAction. Got %s." % action)

        prev_position = self.current_position

        if self.world_map.in_pit_state():
            sample = np.random.multinomial(1, self.init_dist)
            self.current_position = self.positions[np.where(sample > 0)[0]]
        elif self.world_map.in_trap_state():
            pass
        else:
            if(np.random.random() < self.noise):
                perp_dirs = action.get_perpendicular_directions()
                action = perp_dirs[np.random.randint(len(perp_dirs))]

            next_position = action.get_next_position(self.current_position)

            if self.world_map.is_valid_position(next_position):
                self.current_position = next_position

        reward = self.get_reward(
            action, self.pos2state(prev_position), self.current_state)

        return self.current_state, reward

    @property
    def current_position(self):
        return self.world_map.current_position

    @current_position.setter
    def current_position(self, pos):
        try:
            self.world_map.current_position = Position(pos)
        except:
            raise TypeError(
                "Cannot set current position to %s with "
                "type %s." % (pos, type(pos)))

    @property
    def actions(self):
        return GridAction.get_all_actions()

    @property
    def states(self):
        dimension = len(self.positions)
        return [
            GridState(pos, i, dimension) for i, pos
            in enumerate(self.positions)]

    @property
    def n_actions(self):
        return len(self.actions)

    @property
    def n_states(self):
        return len(self.states)

    @property
    def current_state(self):
        return self.pos2state(self.current_position)

    def has_terminal_states(self):
        return self.terminate_on_goal

    def in_terminal_state(self):
        return (
            self.has_terminal_states() and self.world_map.in_terminal_state())

    def in_pit_state(self):
        return self.world_map.is_pit_state(self.current_position)

    def in_puddle_state(self):
        return self.world_map.is_puddle_state(self.current_position)

    def make_T(self):
        """
        Returns a set of transition matrices, one for each action.
        For each a, each row of T[a] sums to 1.
        """
        # local transition probabilities when trying to move straight
        # see fn execute_action for where this is implemented
        # dict index is (S, R, L), whether the location is blocked by a wall.
        # probabilities are (C, S, R, L), probability of transitioning there.
        # C is current position
        s = 1 - self.noise
        n = self.noise / 2.0

        local_transitions = {
            (0, 0, 0): [0.0, s, n, n],
            (0, 0, 1): [n, s, n, 0.0],
            (0, 1, 0): [n, s, 0.0, n],
            (0, 1, 1): [n+n, s, 0.0, 0.0],
            (1, 0, 0): [s, 0.0, n, n],
            (1, 0, 1): [s+n, 0.0, n, 0.0],
            (1, 1, 0): [s+n, 0.0, 0.0, n],
            (1, 1, 1): [1.0, 0.0, 0.0, 0.0]
            }

        positions = self.positions
        n_positions = self.n_states

        state_indices = {
            state: index for index, state in enumerate(positions)}

        T = np.zeros(
            (self.n_actions, n_positions, n_positions))

        for a in self.actions:
            for i, pos in enumerate(positions):
                if pos in self.pit_positions:
                    T[a, i, :] = self.init_dist
                elif pos in self.trap_positions:
                    T[a, i, i] = 1.0
                else:
                    straight = a.get_next_position(pos)
                    right = a.right().get_next_position(pos)
                    left = a.left().get_next_position(pos)

                    key = tuple([
                        not self.world_map.is_valid_position(loc)
                        for loc in (straight, right, left)])

                    probs = local_transitions[key]
                    T[a, i, i] = probs[0]

                    iterator = zip(key, [straight, right, left], probs[1:])
                    for invalid, direction, prob in iterator:
                        if not invalid:
                            T[a, i, state_indices[direction]] = prob

                assert sum(T[a, i, :]) == 1.0

        self._T = T
        self._T.flags.writeable = False

    def make_R(self):
        """
        Returns a |actions| x |states| x |states| reward matrix.
        """

        R = np.zeros((self.n_actions, self.n_states, self.n_states))

        for a in self.actions:
            for s in self.states:
                for s_prime in self.states:
                    R[a, s, s_prime] = self.get_reward(a, s, s_prime)

        self._R = R
        self._R.flags.writeable = False

    def get_reward(self, a, s, s_prime=None):
        if s_prime is None:
            return self.get_expected_reward(a, s)

        if self.world_map.is_goal_state(s_prime):
            return self.goal_reward
        elif self.world_map.is_death_state(s_prime):
            return self.death_reward
        elif self.world_map.is_trap_state(s_prime):
            return self.trap_reward
        elif self.world_map.is_pit_state(s_prime):
            return self.pit_reward
        elif self.world_map.is_puddle_state(s_prime):
            return self.puddle_reward
        else:
            return self.default_reward

    @property
    def init_dist(self):
        if self.init_position:
            init_dist = np.zeros(len(self.positions))
            init_dist[self.positions.index(self.init_position)] = 1.0
        else:
            init_dist = np.array(
                [not self.world_map.is_pit_state(p)
                 and not self.world_map.is_puddle_state(p)
                 and not self.world_map.is_terminal_state(p)
                 for p in self.positions])

            init_dist = init_dist / float(sum(init_dist))

        return init_dist

    def pos2state(self, position):
        return GridState(
            position.position, id=self.positions.index(position),
            dimension=len(self.positions))

    def print_value_function(self, V):
        """
        V is an ndarray indexed by state, mapping to
        an a float giving the value of that state under
        some policy.
        """

        max_length = max(len(str(v)) for v in V)

        value_func = self.world_map.copy()
        value_func = value_func.astype('|S%s' % max_length)

        for i in xrange(self.world_map.world_map.shape[0]):
            for j in xrange(self.world_map.world_map.shape[1]):
                pos = (i, j)
                if pos in self.positions:
                    value_func[pos] = str(V[self.pos2state(pos)])

        return value_func

    def print_deterministic_policy(self, pi):
        """
        pi is an ndarray indexed by state, mapping to
        an integer that corresponds to an action.
        """

        policy = self.world_map.copy()

        for i in xrange(self.world_map.world_map.shape[0]):
            for j in xrange(self.world_map.world_map.shape[1]):
                pos = (i, j)
                if pos in self.positions:
                    policy[pos] = GridAction.symbols[pi[self.pos2state(pos)]]

        return policy




