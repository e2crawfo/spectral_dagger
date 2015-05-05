import numpy as np

from spectral_dagger.grid_world import GridWorld, EgoGridWorld
from spectral_dagger.mdp import UniformRandomPolicy as MDPUniformRandomPolicy
from spectral_dagger.pomdp import UniformRandomPolicy as POUniformRandomPolicy


def test_grid_world():
    env = GridWorld()

    policy = MDPUniformRandomPolicy(env)

    env.sample_trajectory(
        policy, horizon=10, reset=True, display=False)


def test_cliff_world():
    cliff_world = np.array([
        ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x'],
        ['x', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x'],
        ['x', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x'],
        ['x', 'A', 'O', 'O', 'O', 'O', 'O', 'O', 'G', 'x'],
        ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x']]
    )

    env = GridWorld(cliff_world)

    from value_iteration import ValueIteration
    alg = ValueIteration()
    policy = alg.fit(env)
    env.sample_trajectory(
        policy, horizon=50, reset=True, display=False)

    from mdp import UniformRandomPolicy
    env.sample_trajectory(
        UniformRandomPolicy(env), horizon=50, reset=True, display=False)


def test_ego_grid_world():
    env = EgoGridWorld(2)

    policy = POUniformRandomPolicy(env)

    env.sample_trajectory(
        policy, horizon=10, reset=True, display=False)
