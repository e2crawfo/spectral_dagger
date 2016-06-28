import numpy as np
import itertools

from spectral_dagger import process_rng
from spectral_dagger import Action, Observation, State, Space
from spectral_dagger.mdp import MDP, MDPPolicy
from spectral_dagger.pomdp import POMDP
from spectral_dagger.utils import sample_multinomial
from spectral_dagger.utils.geometry import Position, Rectangle


class GridState(State):
    def __init__(self, position, id):
        self.position = np.array(position, copy=True)
        self.position.flags.writeable = False
        super(GridState, self).__init__(id)

    @property
    def y(self):
        return self.position[0]

    @property
    def x(self):
        return self.position[1]

    def __str__(self):
        return "<GridState id: %s, position: (y: %d, x: %d)>" % (
            self.get_id(), self.position[0], self.position[1])

    def __array__(self, dtype=np.float32, copy=True,
                  order='C', subok=True, ndmin=0):
        return self.position.copy()


NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3


class GridAction(Action):
    strings = ['NORTH', 'EAST', 'SOUTH', 'WEST']
    symbols = ['^', '>', 'v', '<']

    string_ids = {item: i for i, item in enumerate(strings)}
    symbol_ids = {item: i for i, item in enumerate(symbols)}

    def __init__(self, direction):
        if isinstance(direction, int):
            id = direction
        elif isinstance(direction, GridAction):
            id = direction.get_id()
        elif isinstance(direction, str):
            if len(direction) == 1:
                id = GridAction.symbol_ids[direction]
            else:
                id = GridAction.string_ids[direction.upper()]
        else:
            raise NotImplementedError(
                "Cannot create a GridAction from object "
                "%s of type %s." % (direction, type(direction)))

        super(GridAction, self).__init__(id)

    def __str__(self):
        return "<GridAction direction: %s>" % GridAction.strings[self.get_id()]

    def left(self):
        return GridAction((self.get_id() - 1) % 4)

    def right(self):
        return GridAction((self.get_id() + 1) % 4)

    def get_perpendicular_directions(self):
        if self == NORTH or self == SOUTH:
            return (GridAction(EAST), GridAction(WEST))
        elif self == EAST or self == WEST:
            return (GridAction(NORTH), GridAction(SOUTH))

    def get_offset(self):
        new_pos = {
            NORTH: [-1, 0],
            EAST: [0, 1],
            SOUTH: [1, 0],
            WEST: [0, -1]}[self.get_id()]
        return np.array(new_pos)

    def get_next_position(self, position):
        return Position(position) + self.get_offset()

    @staticmethod
    def get_all_actions():
        return sorted(
            [GridAction(s) for s in GridAction.strings],
            key=lambda a: a.get_id())


class WorldMap(object):
    default_world_map = np.array([
        ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x'],
        ['x', ' ', ' ', ' ', 'x', ' ', ' ', 'S', ' ', ' ', ' ', ' ', ' ', 'x'],
        ['x', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x', 'G', ' ', 'x'],
        ['x', 'x', 'x', 'x', ' ', ' ', ' ', ' ', ' ', ' ', 'x', ' ', ' ', 'x'],
        ['x', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x', ' ', ' ', 'x'],
        ['x', ' ', ' ', ' ', 'x', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x'],
        ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x']]
    )

    """
    Rules for parsing a map:
        1. First find any special characters like S, G, record their locations,
           and turn them into spaces.
            * "S" specifies agent start location. If not provided, agent start
              location chosen uniformly at random from all floor tiles each
              time the environment is reset.
            * "G" specifies the goal location. For now there has to be exactly
              one of these.
            * "O" (capital O) specifies a pit or a hole. Stepping on such a
              space will cause the agent to suffer a large negative reward,
              and go back to the initial state.
            * "P" specifies a puddle. Stepping on such a
              space will cause the agent to suffer a large negative reward,
              but they will not reset back to the initial state.
            * "D" specifies a "death" square. Stepping on such a
              space will cause the episode to end, and potentially cause
              the agent to suffer a large negative reward.
            * "T" specifies a "trap" square. Stepping on such a
              space will cause the agent to be unable to move for the remainder
              of the episode, and potentially suffer a large negative reward.
        2. All spaces are considered empty space which the agent can occupy.
        3. All other remaining characters are considered walls.
        4. The map must be surrounded by walls. Otherwise, agents will try to
           step off the edge of the map, causing errors.

    """
    START_MARKER = 'S'
    GOAL_MARKER = 'G'
    PIT_MARKER = 'O'
    PUDDLE_MARKER = 'P'
    DEATH_MARKER = 'D'
    TRAP_MARKER = 'T'
    WALL_MARKER = 'x'

    def __init__(self, world_map):
        if world_map is None:
            world_map = WorldMap.default_world_map

        self.world_map = np.array(world_map, copy=True)
        self.parse_map()

    def parse_map(self):
        init_positions = self.get_locations_of(WorldMap.START_MARKER)

        self.bounds = Rectangle(
            top_left=(0.0, 0.0), s=np.array(self.world_map.shape)-1)

        if not init_positions:
            self.init_position = None
        else:
            self.init_position = init_positions[0]

        goal_positions = self.get_locations_of(WorldMap.GOAL_MARKER)
        if not goal_positions:
            raise ValueError(
                "World world_map must contain a `%s` "
                "indicating goal location." % WorldMap.GOAL_MARKER)
        else:
            self.goal_position = goal_positions[0]

        self.pit_positions = self.get_locations_of(WorldMap.PIT_MARKER)
        self.puddle_positions = self.get_locations_of(WorldMap.PUDDLE_MARKER)
        self.death_positions = self.get_locations_of(WorldMap.DEATH_MARKER)
        self.trap_positions = self.get_locations_of(WorldMap.TRAP_MARKER)

        self.positions = self.get_locations_of(' ')
        self.positions.extend(self.pit_positions)
        self.positions.extend(self.puddle_positions)
        self.positions.extend(self.death_positions)
        self.positions.extend(self.trap_positions)
        self.positions.append(self.goal_position)

        if self.init_position:
            self.positions.append(self.init_position)

        self.positions.sort(key=lambda p: tuple(p.position), reverse=False)

        self.current_position = None

    def get_locations_of(self, c):
        """ Returns a list of Position objects, giving the positions where the
        the world_map array takes value `c` """

        positions = np.where(self.world_map == c)
        positions = [Position(i) for i in zip(*positions)]
        return positions

    def is_valid_position(self, pos):
        return pos in self.positions

    def in_terminal_state(self):
        return self.is_terminal_state(self.current_position)

    def in_goal_state(self):
        return self.is_goal_state(self.current_position)

    def in_pit_state(self):
        return self.is_pit_state(self.current_position)

    def in_puddle_state(self):
        return self.is_puddle_state(self.current_position)

    def in_death_state(self):
        return self.is_death_state(self.current_position)

    def in_trap_state(self):
        return self.is_trap_state(self.current_position)

    def is_terminal_state(self, s):
        return self.is_goal_state(s) or self.is_death_state(s)

    def is_goal_state(self, s):
        return Position(s) == self.goal_position

    def is_pit_state(self, s):
        return Position(s) in self.pit_positions

    def is_puddle_state(self, s):
        return Position(s) in self.puddle_positions

    def is_death_state(self, s):
        return Position(s) in self.death_positions

    def is_trap_state(self, s):
        return Position(s) in self.trap_positions

    def __getitem__(self, index):
        if isinstance(index, np.ndarray) and index.size == 2:
            index = tuple(index)
        elif isinstance(index, Position):
            index = tuple(np.array(index))

        return self.world_map[index]

    def __setitem__(self, index, item):
        if isinstance(index, np.ndarray) and index.size == 2:
            index = tuple(index)
        elif isinstance(index, Position):
            index = tuple(np.array(index))

        self.world_map[index] = item

    def array_rep(self):
        env = np.zeros(self.world_map.shape, dtype=np.dtype('S1'))
        for i in xrange(self.world_map.shape[0]):
            for j in xrange(self.world_map.shape[1]):
                if (i, j) == self.init_position:
                    env[i, j] = WorldMap.START_MARKER
                elif (i, j) == self.goal_position:
                    env[i, j] = WorldMap.GOAL_MARKER
                elif (i, j) in self.pit_positions:
                    env[i, j] = WorldMap.PIT_MARKER
                elif (i, j) in self.puddle_positions:
                    env[i, j] = WorldMap.PUDDLE_MARKER
                elif (i, j) in self.death_positions:
                    env[i, j] = WorldMap.DEATH_MARKER
                elif (i, j) in self.trap_positions:
                    env[i, j] = WorldMap.TRAP_MARKER
                else:
                    env[i, j] = str(self.world_map[i, j])

        return env

    def __str__(self):
        env = self.array_rep()

        for i in xrange(self.world_map.shape[0]):
            for j in xrange(self.world_map.shape[1]):
                if (self.current_position is not None
                        and (i, j) == self.current_position):
                    env[i, j] = 'A'

        return '\n'.join([''.join(row) for row in env])


class ColoredWorldMap(WorldMap):
    def __init__(self, n_colors, world_map=None, rng=None):
        assert n_colors >= 1

        if world_map is None:
            world_map = WorldMap.default_world_map
        self.world_map = np.array(world_map, copy=True)

        self.n_colors = n_colors
        self.parse_map(rng)

    def parse_map(self, rng=None):
        n_walls = np.count_nonzero(self.world_map == WorldMap.WALL_MARKER)
        is_wall = self.world_map == WorldMap.WALL_MARKER

        rng = process_rng(rng)
        self.world_map[is_wall] = (
            rng.randint(1, self.n_colors+1, n_walls))

        super(ColoredWorldMap, self).parse_map()

    def get_color_at(self, pos):
        # TODO: designate colors for pits and puddles
        val = self[pos]

        try:
            # A wall
            val = int(val)
            return val
        except:
            # Anything else
            return 0


class GridWorld(MDP):

    DEFAULT_REWARD = 0
    GOAL_REWARD = 10
    PIT_REWARD = -100
    PUDDLE_REWARD = -100
    DEATH_REWARD = -100
    TRAP_REWARD = -1

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

    def has_terminal_states(self):
        return self.terminate_on_goal

    def in_terminal_state(self):
        return (
            self.has_terminal_states() and self.world_map.in_terminal_state())

    @property
    def actions(self):
        return GridAction.get_all_actions()

    @property
    def states(self):
        return [
            GridState(pos, i) for i, pos
            in enumerate(self.positions)]

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
        """ Resets the state of the grid world.

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

                locations = list(locations)

                self.current_position = (
                    locations[self.run_rng.randint(len(locations))])
            else:
                self.current_position = Position(self.init_position)

        else:
            try:
                self.current_position = Position(state)
            except:
                raise ValueError(
                    "GridWorld.reset received invalid starting "
                    "state: %s" % state)

        return self.pos2state(self.current_position)

    def step(self, action):
        """ Update state of the grid world given that ``action`` was taken.

        With probability 1 - self.noise, move in the specified direction. With
        probability self.noise, move in a randomly chosen perpendicular
        direction. If there is a wall in the direction selected by this random
        process, stay in place.

        Returns
        -------
        New state and reward received.

        """
        try:
            action = GridAction(action)
        except:
            raise ValueError(
                "Action for GridWorld must either be GridAction, or "
                "something convertible to a GridAction. Got %s." % action)

        prev_position = self.current_position

        if self.world_map.in_pit_state():
            sample = sample_multinomial(self.init_dist, self.run_rng)
            self.current_position = self.positions[sample]

        elif self.world_map.in_trap_state():
            pass

        else:
            if(self.run_rng.rand() < self.noise):
                perp_dirs = action.get_perpendicular_directions()
                action = self.run_rng.choice(perp_dirs)

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
    def current_state(self):
        return self.pos2state(self.current_position)

    def in_pit_state(self):
        return self.world_map.is_pit_state(self.current_position)

    def in_puddle_state(self):
        return self.world_map.is_puddle_state(self.current_position)

    def make_T(self):
        """ Make transition operator. """

        # local transition probabilities when trying to move straight
        # see fn ``step`` for where this is implemented
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
        """ Make reward operator. """

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
            position.position, id=self.positions.index(position))

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


class GridObservation(Observation):
    def __init__(self, north, east, south, west, n_values):
        """
        n_values is the number of different values we can encounter in
        a single direction. So usually the number of wall colors plus 1
        for non-walls (which are given value 0).
        """

        self.values = [north, east, south, west]
        self.n_values = n_values

        super(GridObservation, self).__init__(self.get_id())

    def get_id(self):
        return sum(
            [d * self.n_values**e for d, e in zip(self.values, range(4))])

    def __str__(self):
        return ("<GridObs id: %d, N: %d, E: %d, "
                "S: %d, W: %d>" % tuple([self.get_id()] + self.values))

    @staticmethod
    def get_all_observations(n_values):
        d = range(n_values)
        observations = [
            GridObservation(*(i + (n_values,)))
            for i in itertools.product(d, d, d, d)]

        return sorted(observations, key=lambda o: o.get_id())


class EgoGridWorld(POMDP):
    """ An Egocentric Grid World.

    Takes in a world map where all the walls are either x's or integers from
    1 to n_colors. For each x, randomly replaces it with a number from
    1 to n_colors. The result is a world map that effectively has colored
    walls.

    """
    def __init__(
            self, n_colors, world_map=None, gamma=0.9, noise=0.1, rng=None):

        self.world_map = ColoredWorldMap(n_colors, world_map, rng=rng)
        self.grid_world = GridWorld(self.world_map, gamma=gamma, noise=noise)

        self.make_O()
        self.reset()

    @property
    def name(self):
        return "EgocentricGridWorld"

    def __str__(self):
        return str(self.grid_world)

    @property
    def actions(self):
        return GridAction.get_all_actions()

    @property
    def observations(self):
        return GridObservation.get_all_observations(self.world_map.n_colors+1)

    @property
    def states(self):
        return self.grid_world.states

    @property
    def mdp(self):
        return self.grid_world

    def reset(self, init_dist=None):
        """ Resets the state of the grid world. """

        if init_dist is not None:
            if len(init_dist) != self.n_states:
                raise ValueError(
                    "Initialization distribution must have number of elements "
                    "equal to the number of states in the POMDP.")

            sample = sample_multinomial(init_dist, self.run_rng)
            self.grid_world.reset(sample)

        else:
            self.grid_world.reset()

    def step(self, action):
        """ Update state of the grid world given that ``action`` was taken.

        Returns
        -------
        New state and reward received.

        """
        state, reward = self.grid_world.step(action)
        obs = self.generate_observation()

        return obs, reward

    def generate_observation(self):
        pos = self.grid_world.current_position

        north = self.world_map.get_color_at(pos + (-1, 0))
        east = self.world_map.get_color_at(pos + (0, 1))
        south = self.world_map.get_color_at(pos + (1, 0))
        west = self.world_map.get_color_at(pos + (0, -1))

        return GridObservation(
            north, east, south, west, self.world_map.n_colors+1)

    def make_O(self):
        """ Make observation operator. """

        temp_state = self.current_state

        O = np.zeros((
            self.n_actions, self.n_states,
            self.n_observations))

        for i in range(self.n_states):
            self.grid_world.reset(i)
            obs = self.generate_observation()
            O[:, i, obs.get_id()] = 1.0

        self.grid_world.reset(temp_state)

        self._O = O
        self._O.flags.writeable = False

    @property
    def O(self):
        return self._O

    @property
    def init_dist(self):
        return self.grid_world.init_dist


class GridKeyboardPolicy(MDPPolicy):
    """ A policy that accepts input from the keyboard using WASD. """

    def __init__(self, mapping=None):
        if mapping is None:
            mapping = {'w': 0, 'd': 1, 's': 2, 'a': 3}

        self.actions = GridAction.get_all_actions()
        self.mapping = mapping

    @property
    def observation_space(self):
        return Space(name="ObsSpace")

    def get_action(self):
        x = raw_input()
        assert len(x) == 1
        return self.mapping[x]
