import numpy as np

import itertools

from mdp import MDP, State, Action
from pomdp import POMDP, Observation
from geometry import Position


class GridState(State):
    def __init__(self, position, id, dimension):
        self.position = np.array(position, copy=True)
        self.position.flags.writeable = False
        super(GridState, self).__init__(id, dimension)

    def as_vector(self):
        return np.array([self.y, self.x])

    @property
    def y(self):
        return self.position[0]

    @property
    def x(self):
        return self.position[1]

    def __str__(self):
        return "<GridState id: %s, position: (y: %d, x: %d), dim: %d>" % (
            self.get_id(), self.position[0], self.position[1], self.dimension)

    def __array__(self):
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

    def __init__(self, dir):
        if isinstance(dir, int):
            id = dir
        elif isinstance(dir, GridAction):
            id = dir.get_id()
        elif isinstance(dir, str):
            if len(dir) == 1:
                id = GridAction.symbol_ids[dir]
            else:
                id = GridAction.string_ids[dir.upper()]
        else:
            raise NotImplementedError(
                "Cannot create a GridAction from object "
                "%s of type %s." % (dir, type(dir)))

        super(GridAction, self).__init__(id)

    def __str__(self):
        return "<GridAction dir: %s>" % GridAction.strings[self.get_id()]

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
        if self.get_id() == NORTH:
            return np.array([-1, 0])
        elif self.get_id() == EAST:
            return np.array([0, 1])
        elif self.get_id() == SOUTH:
            return np.array([1, 0])
        elif self.get_id() == WEST:
            return np.array([0, -1])

    def get_next_position(self, position):
        return Position(position) + self.get_offset()

    @staticmethod
    def get_all_actions():
        return [GridAction(s) for s in GridAction.strings]


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
        2. All spaces are considered empty space which the agent can occupy.
        3. All other remaining characters are considered walls.
        4. The map must be surrounded by walls. Otherwise, agents will try to
           step off the edge of the map, causing errors.
    """

    START_MARKER = 'S'
    GOAL_MARKER = 'G'
    PIT_MARKER = 'O'
    PUDDLE_MARKER = 'P'
    WALL_MARKER = 'x'

    def __init__(self, world_map):
        if world_map is None:
            world_map = WorldMap.default_world_map

        self.world_map = np.array(world_map, copy=True)
        self.parse_map()

    def parse_map(self):
        init_positions = self.get_locations_of(WorldMap.START_MARKER)

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

        self.positions = self.get_locations_of(' ')
        self.positions.extend(self.pit_positions)
        self.positions.extend(self.puddle_positions)
        self.positions.append(self.goal_position)

        if self.init_position:
            self.positions.append(self.init_position)

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

    def in_pit_state(self):
        return self.is_pit_state(self.current_position)

    def in_puddle_state(self):
        return self.is_puddle_state(self.current_position)

    def is_terminal_state(self, s):
        return Position(s) == self.goal_position

    def is_pit_state(self, s):
        return Position(s) in self.pit_positions

    def is_puddle_state(self, s):
        return Position(s) in self.puddle_positions

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

    def __str__(self):
        w = ''
        for i in xrange(self.world_map.shape[0]):
            for j in xrange(self.world_map.shape[1]):
                if (self.current_position is not None
                        and (i, j) == self.current_position):
                    w += 'A'
                elif (i, j) == self.init_position:
                    w += WorldMap.START_MARKER
                elif (i, j) == self.goal_position:
                    w += WorldMap.GOAL_MARKER
                elif (i, j) in self.pit_positions:
                    w += WorldMap.PIT_MARKER
                elif (i, j) in self.puddle_positions:
                    w += WorldMap.PUDDLE_MARKER
                else:
                    w += str(self.world_map[i][j])

            if i < self.world_map.shape[0] - 1:
                w += '\n'

        return w


class ColoredWorldMap(WorldMap):
    def __init__(self, n_colors, world_map=None):
        assert n_colors >= 1

        if world_map is None:
            world_map = WorldMap.default_world_map
        self.world_map = np.array(world_map, copy=True)

        self.n_colors = n_colors
        self.parse_map()

    def parse_map(self):
        n_walls = np.count_nonzero(self.world_map == WorldMap.WALL_MARKER)
        is_wall = self.world_map == WorldMap.WALL_MARKER
        self.world_map[is_wall] = (
            np.random.randint(1, self.n_colors+1, n_walls))

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

    GOAL_REWARD = 10
    PIT_REWARD = -100
    PUDDLE_REWARD = -100

    def __init__(self, world_map=None, gamma=0.9, noise=0.1, rewards=None):
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
        self.positions = self.world_map.positions

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
        self.puddle_reward = rewards.get('puddle', GridWorld.PUDDLE_REWARD)
        self.pit_reward = rewards.get('pit', GridWorld.PIT_REWARD)
        self.goal_reward = rewards.get('goal', GridWorld.GOAL_REWARD)

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
        return True

    def in_terminal_state(self):
        return self.world_map.in_terminal_state()

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

        if self.world_map.is_terminal_state(s_prime):
            return self.goal_reward
        elif self.world_map.is_pit_state(s_prime):
            return self.pit_reward
        elif self.world_map.is_puddle_state(s_prime):
            return self.puddle_reward
        else:
            return 0

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

        return observations


class EgoGridWorld(POMDP):
    """
    For Egocentric Grid World.

    Takes in a world map where all the walls are either x's or integers from
    1 to n_colors. For each x, randomly replaces it with a number from
    1 to n_colors. The result is a world map that effectively has colored
    walls.
    """

    def __init__(self, n_colors, world_map=None, gamma=0.9, noise=0.1):
        self.world_map = ColoredWorldMap(n_colors, world_map)
        self.grid_world = GridWorld(self.world_map, gamma=gamma, noise=noise)

        self.make_O()
        self.reset()

    @property
    def name(self):
        return "EgocentricGridWorld"

    def __str__(self):
        return str(self.grid_world)

    @property
    def mdp(self):
        return self.grid_world

    def reset(self, init_dist=None):
        """
        Can specify an initial distribution, or supply None to have it use
        its internal intial distribution.
        """
        if init_dist is not None:
            if len(init_dist) != self.n_states:
                raise ValueError(
                    "Initialization distribution must have number of elements "
                    "equal to the number of states in the POMDP.")

            sample = np.random.multinomial(1, init_dist)
            self.grid_world.reset(np.where(sample > 0)[0][0])

        else:
            self.grid_world.reset()

    def execute_action(self, action):
        """
        Play the given action.

        Returns the resulting observation and reward.
        """

        state, reward = self.grid_world.execute_action(action)
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

    @property
    def actions(self):
        return GridAction.get_all_actions()

    @property
    def observations(self):
        return GridObservation.get_all_observations(self.world_map.n_colors+1)

    @property
    def states(self):
        return self.grid_world.states

    def make_O(self):
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
