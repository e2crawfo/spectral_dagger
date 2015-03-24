import numpy as np

import time
import itertools

from pomdp import POMDP, State, Action, Observation

NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3


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


class GridObservation(Observation):
    def __init__(self, north, east, south, west, num_values):
        """max_value is the maximum value of each field"""
        self.values = [north, east, south, west]
        self.num_values = num_values

        super(GridObservation, self).__init__(id)

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


class GridWorld(POMDP):

    default_world = np.array([
        ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x'],
        ['x', ' ', ' ', ' ', 'x', ' ', ' ', 'A', ' ', ' ', ' ', ' ', ' ', 'x'],
        ['x', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x', 'G', ' ', 'x'],
        ['x', 'x', 'x', 'x', ' ', ' ', ' ', ' ', ' ', ' ', 'x', ' ', ' ', 'x'],
        ['x', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x', ' ', ' ', 'x'],
        ['x', ' ', ' ', ' ', 'x', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x'],
        ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x']]
    )

    def __init__(self, world=None):
        if world is None:
            world = GridWorld.default_world

        self.world = world.copy()

        if 'A' not in self.world:
            self.init_position = None
        else:
            init_position = np.where(world == 'A')
            self.init_position = init_position[0][0], init_position[1][0]
            self.world[self.init_position] = ' '

        if 'G' not in self.world:
            raise ValueError(
                "World map must have a G indicating goal location.")

        goal_position = np.where(world == 'G')
        self.goal_position = goal_position[0][0], goal_position[1][0]

        self.world[self.goal_position] = ' '

        self.positions = zip(*np.where(self.world == ' '))

        self.reset()

    @property
    def name(self):
        return "GridWorld"

    @property
    def num_actions(self):
        return len(self.actions)

    @property
    def actions(self):
        return GridAction.get_all_actions()

    @property
    def num_observations(self):
        return len(self.observations)

    @property
    def observations(self):
        return GridObservation.get_all_observations(2)

    @property
    def num_states(self):
        return len(self.states)

    @property
    def states(self):
        return [
            GridState(pos, i) for i, pos in enumerate(self.positions)]

    @property
    def init_dist(self):
        if self.init_position:
            init_dist = np.zeros(len(self.states))
            init_dist[self.positions.index(self.init_position)] = 1.0
        else:
            init_dist = np.ones(len(self.states)) / self.num_states

        return init_dist

    @property
    def state(self):
        return self.pos2state(self.current_position)

    def pos2state(self, position):
        return GridState(position, self.positions.index(position))

    def reset(self, init_dist=None):
        if init_dist is not None:
            if len(init_dist) != self.num_states:
                raise ValueError(
                    "Initialization distribution must have number of elements "
                    "equal to the number of states in the POMDP.")

            sample = np.random.multinomial(1, init_dist)
            self.current_position = self.positions[np.where(sample > 0)[0][0]]

        elif self.init_position:
            self.current_position = self.init_position

        else:
            locations = zip(*np.where(self.world == ' '))
            self.current_position = locations[
                np.random.randint(len(locations))]

    def in_terminal_state(self):
        return self.current_position == self.goal_position

    def execute_action(self, action):
        """
        With probability 0.8, move in the specified direction. With
        probability 0.2, move in a randomly chosen perpendicular direction.
        If there is a wall in the direction selected by this random process,
        stay in place.
        """
        if(np.random.random() < 0.2):
            perp_dirs = action.get_perpendicular_directions()
            action = perp_dirs[np.random.randint(len(perp_dirs))]

        next_position = action.get_next_position(self.current_position)

        if self.world[next_position] == ' ':
            self.current_position = next_position

        self.observation = self.generate_observation()
        self.reward = self.get_reward(self.current_position, None)

        return self.observation, self.reward

    def get_reward(self, state, action):
        if isinstance(state, GridState):
            in_goal_state = state.position == self.goal_position
        else:
            in_goal_state = state == self.goal_position

        return 1.0 if in_goal_state else 0.0

    def generate_observation(self):
        y_pos, x_pos = self.current_position

        north = self.get_obs(y_pos-1, x_pos)
        east = self.get_obs(y_pos, x_pos+1)
        south = self.get_obs(y_pos+1, x_pos)
        west = self.get_obs(y_pos, x_pos-1)

        return GridObservation(north, east, south, west, 2)

    def get_obs(self, y, x):
        if self.world[y][x] == 'x':
            return 1
        else:
            return 0

    def get_reward_op(self):
        """
        Returns a |actions| x |states| reward matrix.
        """
        R = np.zeros((self.num_actions, self.num_states))

        for a in self.actions:
            for s in self.states:
                R[a, s] = self.get_reward(s, a)

        return R

    def get_transition_op(self):
        """
        Returns a set of transition matrices, one for each action.
        For each a, each row of T[a] sums to 1.
        """

        # local transition probabilities when trying to move straight
        # see fn execute_action for where this is implemented
        # index is (S, R, L), whether the location is blocked
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
                    self.world[space] != ' '
                    for space in (straight, right, left)])

                probs = local_transitions[key]
                T[a, i, i] = probs[0]

                iterator = zip(key, [straight, right, left], probs[1:])
                for invalid, direction, prob in iterator:
                    if not invalid:
                        T[a, i, state_indices[direction]] = prob

                assert sum(T[a, i, :]) == 1.0

        return T

    def get_observation_op(self):
        temp = self.current_position
        valid_states = zip(*np.where(self.world == ' '))
        num_valid_states = len(valid_states)

        O = np.zeros((
            self.num_actions, num_valid_states,
            self.num_observations))

        for i, state in enumerate(valid_states):
            self.current_position = state
            obs = self.get_current_observation()
            O[:, i, obs.get_id()] = 1.0

        self.current_position = temp

        return O

    def __str__(self):
        w = ''
        for i in xrange(self.world.shape[0]):
            for j in xrange(self.world.shape[1]):
                if (i, j) == self.current_position:
                    w += 'A'
                elif (i, j) == self.init_position:
                    w += 'I'
                elif (i, j) == self.goal_position:
                    w += 'G'
                else:
                    w += str(self.world[i][j])

            w += '\n'

        return w


class ColoredGridWorld(GridWorld):
    def __init__(self, num_colors, world=None):
        super(ColoredGridWorld, self).__init__(world)

        self.num_colors = num_colors

        num_walls = np.count_nonzero(self.world == 'x')
        self.world[self.world == 'x'] = np.random.randint(
            1, num_colors+1, num_walls)

        self.reset()

    @property
    def num_observations(self):
        return (self.num_colors + 1)**4

    @property
    def observations(self):
        return GridObservation.get_all_observations(self.num_colors+1)

    def get_current_observation(self):
        y_pos, x_pos = self.current_position

        north = self.get_obs(y_pos-1, x_pos)
        east = self.get_obs(y_pos, x_pos+1)
        south = self.get_obs(y_pos+1, x_pos)
        west = self.get_obs(y_pos, x_pos-1)

        return GridObservation(north, east, south, west, self.num_colors+1)

    def get_obs(self, y, x):
        val = self.world[y][x]
        try:
            # A wall
            val = int(val)
            return val
        except:
            # Anything else
            return 0

if __name__ == "__main__":
    world = ColoredGridWorld(2)
    # world = GridWorld()

    print str(world)

    num_steps = 10

    for i in range(num_steps):
        o, r = world.execute_action(GridAction(np.random.randint(4)))
        time.sleep(0.2)

        # Clear screen
        print(chr(27) + "[2J")

        print str(world)
        print o

    T = world.get_transition_op()
    O = world.get_observation_op()
