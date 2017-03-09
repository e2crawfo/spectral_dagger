import numpy as np

from spectral_dagger import Space
from spectral_dagger.envs.grid_world import GridWorld, WorldMap
from spectral_dagger.envs.grid_world import GridAction, GridState
from spectral_dagger.utils.geometry import Rectangle, Circle, Position


class ContinuousWorldMap(WorldMap):
    def __init__(self, world_map, region_radius=0.5):
        if world_map is None:
            world_map = WorldMap.default_world_map

        self.world_map = np.array(world_map, copy=True)
        self.region_radius = region_radius

        self.parse_map()

    def parse_map(self):
        super(ContinuousWorldMap, self).parse_map()

        # TODO: make this work for colored case
        self.walls = [
            Rectangle((1, 1), centre=c, closed=True)
            for c in self.get_locations_of(WorldMap.WALL_MARKER)]

        self.goal_region = Circle(
            self.region_radius, self.goal_position)

        self.puddle_regions = [
            Circle(self.region_radius, p) for p in self.puddle_positions]
        self.pit_regions = [
            Circle(self.region_radius, p) for p in self.pit_positions]
        self.death_regions = [
            Circle(self.region_radius, p) for p in self.death_positions]
        self.trap_regions = [
            Circle(self.region_radius, p) for p in self.trap_positions]

        env_shape = [int(i) for i in self.bounds.s]
        self.lattice = np.meshgrid(
            *[list(range(i+1)) for i in env_shape], indexing='ij')
        self.lattice = list(zip(*[a.flatten() for a in self.lattice]))
        self.lattice = [Rectangle((1, 1), centre=c) for c in self.lattice]

    def is_valid_position(self, pos):
        return pos in self.bounds and not any([pos in r for r in self.walls])

    def is_goal_state(self, s):
        s = s.position if isinstance(s, GridState) else s
        s = Position(s)
        return s in self.goal_region

    def is_pit_state(self, s):
        s = s.position if isinstance(s, GridState) else s
        s = Position(s)
        return any([s in p for p in self.pit_regions])

    def is_puddle_state(self, s):
        s = s.position if isinstance(s, GridState) else s
        s = Position(s)
        return any([s in p for p in self.puddle_regions])

    def is_death_state(self, s):
        s = s.position if isinstance(s, GridState) else s
        s = Position(s)
        return any([s in p for p in self.death_regions])

    def is_trap_state(self, s):
        s = s.position if isinstance(s, GridState) else s
        s = Position(s)
        return any([s in p for p in self.trap_regions])

    def __str__(self):
        env = self.array_rep()
        agent_in_square = [self.current_position in r for r in self.lattice]
        assert sum(agent_in_square) == 1
        index = agent_in_square.index(1)
        env[int(index / env.shape[1]), int(index % env.shape[1])] = 'A'

        return '\n'.join([''.join(row) for row in env])


class ContinuousGridWorld(GridWorld):
    """
    y gives vertical extent, and increases downward. The centre of the
    top-left cell of the environment is the origin. x gives the horizontal
    extent, and increases rightward. Can sort of think of it as the same
    indexing scheme that matrices use in math (except its continuous now).
    """

    def __init__(
            self, world_map=None, gamma=0.9, speed=0.1,
            noise_std=0.01, rewards=None, terminate_on_goal=True):

        self.gamma = gamma
        self.speed = speed
        self.noise_std = noise_std

        self.world_map = ContinuousWorldMap(world_map)

        self.init_position = self.world_map.init_position
        self.goal_position = self.world_map.goal_position
        self.pit_positions = self.world_map.pit_positions
        self.puddle_positions = self.world_map.puddle_positions
        self.positions = self.world_map.positions
        self.terminate_on_goal = terminate_on_goal

        self.set_rewards(rewards)

        self.reset()

    @property
    def name(self):
        return "ContinuousGridWorld"

    def observation_space(self):
        return Space([(-np.inf, np.inf)] * 2, "CtsObsSpace")

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
        elif state is None:
            if self.init_position is None:
                self.current_position = Position(
                    np.random.random(2) * self.world_map.bounds.s)

                while not self.in_valid_start_state():
                    self.current_position = Position(
                        np.random.random(2) * self.world_map.bounds.s)
            else:
                self.current_position = self.init_position
        else:
            try:
                self.current_position = state
            except:
                raise ValueError(
                    "ContinuousGridWorld.reset expected GridState or "
                    "2-dimensional ndarray or None, received %s" % state)

        return self.current_position

    def in_valid_start_state(self):
        return (
            self.world_map.is_valid_position(self.current_position) and
            not self.in_pit_state() and
            not self.in_puddle_state() and
            not self.in_terminal_state())

    def step(self, action):
        action = GridAction(action)

        if self.world_map.in_pit_state():
            prev_position = self.current_position
            self.current_position = self.init_position
        else:
            d_position = self.speed * action.get_offset()
            next_position = self.current_position + d_position

            if self.noise_std > 0:
                next_position += np.random.normal(0, self.noise_std, 2)

            prev_position = self.current_position

            self.current_position = (
                next_position
                if self.world_map.is_valid_position(next_position)
                else self.current_position)

        reward = self.get_reward(
            action, prev_position, self.current_position)

        return self.current_position, reward

    @property
    def current_state(self):
        return self.current_position

    def __str__(self):
        s = "State: %s\n" % self.current_position
        s += str(self.world_map) + '\n'
        s += "In goal: %s\n" % self.in_terminal_state()
        s += "In pit: %s\n" % self.in_pit_state()
        s += "In puddle: %s" % self.in_puddle_state()
        return s

    def __repr__(self):
        return str(self)

    def is_continuous(self):
        return True

    @property
    def n_states(self):
        raise ValueError(
            "Cannot call n_states on environment "
            "with continuous state space.")

    @property
    def T(self):
        raise ValueError(
            "Cannot get transition operator of environment "
            "with continuous state space.")

    @property
    def R(self):
        raise ValueError(
            "Cannot get reward operator of environment "
            "with continuous state space.")
