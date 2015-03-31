import numpy as np

import itertools

from mdp import MDP, State, Action
from pomdp import POMDP, Observation


class GridState(State):
    def __init__(self, position, id):
        self.position = position
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


NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3


class GridAction(Action):
    strings = ['NORTH', 'EAST', 'SOUTH', 'WEST']
    ids = {item: i for i, item in enumerate(strings)}

    def __init__(self, dir):
        if isinstance(dir, int):
            id = dir
        else:
            id = GridAction.ids[dir.upper()]

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

    def get_next_position(self, position):
        """
        Returns a pair of ints (y, x).
        """

        if self.get_id() == NORTH:
            return (position[0]-1, position[1])
        elif self.get_id() == EAST:
            return (position[0], position[1]+1)
        elif self.get_id() == SOUTH:
            return (position[0]+1, position[1])
        elif self.get_id() == WEST:
            return (position[0], position[1]-1)

    @staticmethod
    def get_all_actions():
        return [GridAction(s) for s in GridAction.strings]


class GridWorld(MDP):

    default_world_map = np.array([
        ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x'],
        ['x', ' ', ' ', ' ', 'x', ' ', ' ', 'A', ' ', ' ', ' ', ' ', ' ', 'x'],
        ['x', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x', 'G', ' ', 'x'],
        ['x', 'x', 'x', 'x', ' ', ' ', ' ', ' ', ' ', ' ', 'x', ' ', ' ', 'x'],
        ['x', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x', ' ', ' ', 'x'],
        ['x', ' ', ' ', ' ', 'x', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x'],
        ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x']]
    )

    """
    Rules for parsing a map:
        1. First find any special characters like A, G, record their locations,
           and turn them into spaces.
            * "A" specifies agent start location. If not provided, agent start
              location chosen uniformly at random from all floor tiles each
              time the environment is reset.
            * "G" specifies the goal location. For now there has to be exactly
              one of these.
        2. All spaces are considered empty space which the agent can occupy.
        3. All other remaining characters are considered walls.
        4. The map must be surrounded by walls. Otherwise, agents will try to
           step off the edge of the map, causing errors.
    """

    def __init__(self, world_map=None, gamma=1.0):
        self.gamma = gamma

        if world_map is None:
            world_map = GridWorld.default_world_map

        self.world_map = world_map.copy()

        if 'A' not in self.world_map:
            self.init_position = None
        else:
            init_position = np.where(world_map == 'A')
            self.init_position = init_position[0][0], init_position[1][0]
            self.world_map[self.init_position] = ' '

        if 'G' not in self.world_map:
            raise ValueError(
                "World world_map must have a G indicating goal location.")

        goal_position = np.where(world_map == 'G')
        self.goal_position = goal_position[0][0], goal_position[1][0]

        self.world_map[self.goal_position] = ' '

        self.positions = zip(*np.where(self.world_map == ' '))

        self.reset()

    @property
    def name(self):
        return "GridWorld"

    def __str__(self):
        w = ''
        for i in xrange(self.world_map.shape[0]):
            for j in xrange(self.world_map.shape[1]):
                if (i, j) == self.current_position:
                    w += 'A'
                elif (i, j) == self.init_position:
                    w += 'I'
                elif (i, j) == self.goal_position:
                    w += 'G'
                else:
                    w += str(self.world_map[i][j])

            w += '\n'

        return w

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
            self.current_position = state.position
        elif isinstance(state, int):
            self.current_position = self.positions[state]
        elif state is None:
            if self.init_position is None:
                locations = zip(*np.where(self.world_map == ' '))
                self.current_position = locations[
                    np.random.randint(len(locations))]
            else:
                self.current_position = self.init_position

        else:
            raise ValueError(
                "GridWorld.reset expected GridState or int"
                "or None, received %s" % state)

    def execute_action(self, action):
        """
        With probability 0.8, move in the specified direction. With
        probability 0.2, move in a randomly chosen perpendicular direction.
        If there is a wall in the direction selected by this random process,
        stay in place.

        Returns the next state and the reward.
        """
        if not isinstance(action, GridAction):
            try:
                action = GridAction(action)
            except:
                raise ValueError(
                    "Action for GridWorld must either be GridAction, or "
                    "something convertible to a GridAction. Got %s." % action)

        if(np.random.random() < 0.2):
            perp_dirs = action.get_perpendicular_directions()
            action = perp_dirs[np.random.randint(len(perp_dirs))]

        next_position = action.get_next_position(self.current_position)

        if self.world_map[next_position] == ' ':
            self.current_position = next_position

        reward = self.get_reward(action, self.state)

        return self.state, reward

    @property
    def actions(self):
        return GridAction.get_all_actions()

    @property
    def states(self):
        return [
            GridState(pos, i) for i, pos in enumerate(self.positions)]

    @property
    def num_actions(self):
        return len(self.actions)

    @property
    def num_states(self):
        return len(self.states)

    @property
    def state(self):
        return self.pos2state(self.current_position)

    @property
    def T(self):
        """
        Returns a set of transition matrices, one for each action.
        For each a, each row of T[a] sums to 1.
        """

        # local transition probabilities when trying to move straight
        # see fn execute_action for where this is implemented
        # dict index is (S, R, L), whether the location is blocked by a wall.
        # probabilities are (C, S, R, L), probability of transitioning there.
        # C is current position
        local_transitions = {
            (0, 0, 0): [0.0, 0.8, 0.1, 0.1],
            (0, 0, 1): [0.1, 0.8, 0.1, 0.0],
            (0, 1, 0): [0.1, 0.8, 0.0, 0.1],
            (0, 1, 1): [0.2, 0.8, 0.0, 0.0],
            (1, 0, 0): [0.8, 0.0, 0.1, 0.1],
            (1, 0, 1): [0.9, 0.0, 0.1, 0.0],
            (1, 1, 0): [0.9, 0.0, 0.0, 0.1],
            (1, 1, 1): [1.0, 0.0, 0.0, 0.0]
            }

        positions = self.positions
        num_positions = self.num_states

        state_indices = {
            state: index for index, state in enumerate(positions)}

        T = np.zeros(
            (self.num_actions, num_positions, num_positions))

        for a in self.actions:
            for i, pos in enumerate(positions):
                straight = a.get_next_position(pos)
                right = a.right().get_next_position(pos)
                left = a.left().get_next_position(pos)

                key = tuple([
                    self.world_map[loc] != ' '
                    for loc in (straight, right, left)])

                probs = local_transitions[key]
                T[a, i, i] = probs[0]

                iterator = zip(key, [straight, right, left], probs[1:])
                for invalid, direction, prob in iterator:
                    if not invalid:
                        T[a, i, state_indices[direction]] = prob

                assert sum(T[a, i, :]) == 1.0

        return T

    @property
    def R(self):
        """
        Returns a |actions| x |states| reward matrix.
        """
        R = np.zeros((self.num_actions, self.num_states))

        for a in self.actions:
            for s in self.states:
                R[a, s] = self.get_reward(a, s)

        return R

    def get_reward(self, action, state):
        if isinstance(state, GridState):
            in_goal_state = state.position == self.goal_position
        else:
            in_goal_state = state == self.goal_position

        return 1.0 if in_goal_state else 0.0

    @property
    def init_dist(self):
        if self.init_position:
            init_dist = np.zeros(len(self.states))
            init_dist[self.positions.index(self.init_position)] = 1.0
        else:
            init_dist = np.ones(len(self.states)) / self.num_states

        return init_dist

    def pos2state(self, position):
        return GridState(position, self.positions.index(position))


class GridObservation(Observation):
    def __init__(self, north, east, south, west, num_values):
        """
        num_values is the number of different values we can encounter in
        a single direction. So usually the number of wall colors plus 1
        for non-walls (which are given value 0).
        """

        self.values = [north, east, south, west]
        self.num_values = num_values

        super(GridObservation, self).__init__(self.get_id())

    def get_id(self):
        return sum(
            [d * self.num_values**e for d, e in zip(self.values, range(4))])

    def __str__(self):
        return ("<GridObs id: %d, N: %d, E: %d, "
                "S: %d, W: %d>" % tuple([self.get_id()] + self.values))

    @staticmethod
    def get_all_observations(num_values):
        d = range(num_values)
        observations = [
            GridObservation(*(i + (num_values,)))
            for i in itertools.product(d, d, d, d)]

        return observations


class EgoGridWorld(POMDP):
    """
    For Egocentric Grid World.

    Takes in a world map where all the walls are either x's or integers from
    1 to num_colors. For each x, randomly replaces it with a number from
    1 to num_colors. The result is a world map that effectively has colored
    walls.
    """

    def __init__(self, num_colors, world_map=None, gamma=1.0):
        self.gamma = gamma

        if world_map is None:
            world_map = GridWorld.default_world_map.copy()

        self.num_colors = num_colors

        num_walls = np.count_nonzero(world_map == 'x')
        world_map[world_map == 'x'] = np.random.randint(
            1, num_colors+1, num_walls)

        self.grid_world = GridWorld(world_map)

        self.world_map = self.grid_world.world_map

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
            if len(init_dist) != self.num_states:
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
        y_pos, x_pos = self.grid_world.current_position

        north = self.get_obs_at(y_pos-1, x_pos)
        east = self.get_obs_at(y_pos, x_pos+1)
        south = self.get_obs_at(y_pos+1, x_pos)
        west = self.get_obs_at(y_pos, x_pos-1)

        return GridObservation(north, east, south, west, self.num_colors+1)

    def get_obs_at(self, y, x):
        val = self.world_map[y][x]
        try:
            # A wall
            val = int(val)
            return val
        except:
            # Anything else
            return 0

    @property
    def actions(self):
        return GridAction.get_all_actions()

    @property
    def observations(self):
        return GridObservation.get_all_observations(self.num_colors+1)

    @property
    def states(self):
        return self.grid_world.states

    @property
    def state(self):
        return self.grid_world.state

    @property
    def T(self):
        return self.grid_world.T

    @property
    def O(self):
        temp_state = self.state

        O = np.zeros((
            self.num_actions, self.num_states,
            self.num_observations))

        for i in range(self.num_states):
            self.grid_world.reset(i)
            obs = self.generate_observation()
            O[:, i, obs.get_id()] = 1.0

        self.grid_world.reset(temp_state)

        return O

    @property
    def R(self):
        return self.grid_world.copy()

    def get_reward(self, action, state):
        return self.mdp.get_reward(action, state)

    @property
    def init_dist(self):
        return self.grid_world.init_dist


def test_grid_world():
    from policy import MDPPolicy

    env = GridWorld()

    policy = MDPPolicy()
    policy.fit(env)

    print str(env)

    horizon = 10

    trajectory = env.sample_trajectory(
        policy, horizon, reset=True, display=True)

    print trajectory


def test_ego_grid_world():
    from policy import POMDPPolicy

    env = EgoGridWorld(2)

    policy = POMDPPolicy()
    policy.fit(env)

    print str(env)

    horizon = 10

    trajectory = env.sample_trajectory(
        policy, horizon, reset=True, display=True)

    print trajectory


if __name__ == "__main__":
    test_grid_world()
    test_ego_grid_world()
