import numpy as np

from spectral_dagger.grid_world import GridWorld, EgoGridWorld
from spectral_dagger.mdp import UniformRandomPolicy as MDPUniformRandomPolicy
from spectral_dagger.pomdp import UniformRandomPolicy as POUniformRandomPolicy


def test_parse_map(display=False):
    """
    Make sure that the map is parsed properly and
    that all actions have the desired effects.
    """

    dummy_map = np.array([
        ['x', 'x', 'x', 'x'],
        ['x', ' ', 'G', 'x'],
        ['x', ' ', 'P', 'x'],
        ['x', 'S', 'O', 'x'],
        ['x', 'x', 'x', 'x']])

    dummy_world = GridWorld(dummy_map, noise=0)

    ground_truth = 'xxxx\nx Gx\nx Px\nxAOx\nxxxx'
    assert str(dummy_world) == ground_truth

    dummy_world.execute_action('NORTH')

    ground_truth = 'xxxx\nx Gx\nxAPx\nxSOx\nxxxx'
    assert str(dummy_world) == ground_truth

    dummy_world.execute_action('NORTH')

    ground_truth = 'xxxx\nxAGx\nx Px\nxSOx\nxxxx'
    assert str(dummy_world) == ground_truth

    dummy_world.execute_action('NORTH')
    assert str(dummy_world) == ground_truth

    dummy_world.execute_action('EAST')
    ground_truth = 'xxxx\nx Ax\nx Px\nxSOx\nxxxx'
    assert str(dummy_world) == ground_truth
    assert dummy_world.in_terminal_state()

    dummy_world.execute_action('SOUTH')
    ground_truth = 'xxxx\nx Gx\nx Ax\nxSOx\nxxxx'
    assert str(dummy_world) == ground_truth
    assert dummy_world.in_puddle_state()

    dummy_world.execute_action('WEST')
    ground_truth = 'xxxx\nx Gx\nxAPx\nxSOx\nxxxx'
    assert str(dummy_world) == ground_truth

    dummy_world.execute_action('SOUTH')
    ground_truth = 'xxxx\nx Gx\nx Px\nxAOx\nxxxx'
    assert str(dummy_world) == ground_truth

    dummy_world.execute_action('EAST')
    ground_truth = 'xxxx\nx Gx\nx Px\nxSAx\nxxxx'
    assert str(dummy_world) == ground_truth
    assert dummy_world.in_pit_state()

    dummy_world.execute_action('EAST')
    ground_truth = 'xxxx\nx Gx\nx Px\nxAOx\nxxxx'
    assert str(dummy_world) == ground_truth

    assert not dummy_world.in_pit_state()
    assert not dummy_world.in_puddle_state()
    assert not dummy_world.in_terminal_state()


def test_grid_world(display=False):
    env = GridWorld()

    policy = MDPUniformRandomPolicy(env)

    env.sample_trajectory(
        policy, horizon=10, reset=True, display=display)


def test_cliff_world(display=False):
    cliff_world = np.array([
        ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x'],
        ['x', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x'],
        ['x', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x'],
        ['x', 'S', 'O', 'O', 'O', 'O', 'O', 'O', 'G', 'x'],
        ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x']]
    )

    env = GridWorld(cliff_world)

    from value_iteration import ValueIteration
    alg = ValueIteration()
    policy = alg.fit(env)
    env.sample_trajectory(
        policy, horizon=50, reset=True, display=display)

    from mdp import UniformRandomPolicy
    env.sample_trajectory(
        UniformRandomPolicy(env), horizon=50, reset=True, display=display)


def test_ego_grid_world(display=False):
    env = EgoGridWorld(2)

    policy = POUniformRandomPolicy(env)

    env.sample_trajectory(
        policy, horizon=10, reset=True, display=display)
