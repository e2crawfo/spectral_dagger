import numpy as np

from spectral_dagger.tests.conftest import make_test_display
from spectral_dagger.envs import GridWorld, EgoGridWorld
from spectral_dagger.envs import ContinuousGridWorld
from spectral_dagger.mdp import ValueIteration, UniformRandomPolicy


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

    dummy_world.step('NORTH')

    ground_truth = 'xxxx\nx Gx\nxAPx\nxSOx\nxxxx'
    assert str(dummy_world) == ground_truth

    dummy_world.step('NORTH')

    ground_truth = 'xxxx\nxAGx\nx Px\nxSOx\nxxxx'
    assert str(dummy_world) == ground_truth

    dummy_world.step('NORTH')
    assert str(dummy_world) == ground_truth

    dummy_world.step('EAST')
    ground_truth = 'xxxx\nx Ax\nx Px\nxSOx\nxxxx'
    assert str(dummy_world) == ground_truth
    assert dummy_world.in_terminal_state()

    dummy_world.step('SOUTH')
    ground_truth = 'xxxx\nx Gx\nx Ax\nxSOx\nxxxx'
    assert str(dummy_world) == ground_truth
    assert dummy_world.in_puddle_state()

    dummy_world.step('WEST')
    ground_truth = 'xxxx\nx Gx\nxAPx\nxSOx\nxxxx'
    assert str(dummy_world) == ground_truth

    dummy_world.step('SOUTH')
    ground_truth = 'xxxx\nx Gx\nx Px\nxAOx\nxxxx'
    assert str(dummy_world) == ground_truth

    dummy_world.step('EAST')
    ground_truth = 'xxxx\nx Gx\nx Px\nxSAx\nxxxx'
    assert str(dummy_world) == ground_truth
    assert dummy_world.in_pit_state()

    dummy_world.step('EAST')
    ground_truth = 'xxxx\nx Gx\nx Px\nxAOx\nxxxx'
    assert str(dummy_world) == ground_truth

    assert not dummy_world.in_pit_state()
    assert not dummy_world.in_puddle_state()
    assert not dummy_world.in_terminal_state()


def test_grid_world(display=False):
    display_hook = make_test_display(display)

    env = GridWorld()
    policy = UniformRandomPolicy(env.actions)

    env.sample_episode(policy, horizon=10, hook=display_hook)


def test_cliff_world(display=False):
    display_hook = make_test_display(display)

    cliff_world = np.array([
        ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x'],
        ['x', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x'],
        ['x', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x'],
        ['x', 'S', 'O', 'O', 'O', 'O', 'O', 'O', 'G', 'x'],
        ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x']]
    )

    env = GridWorld(cliff_world)

    alg = ValueIteration()
    policy = alg.fit(env)
    env.sample_episode(policy, horizon=50, hook=display_hook)

    env.sample_episode(
        UniformRandomPolicy(env.actions), horizon=50, hook=display_hook)


def test_ego_grid_world(display=False):
    display_hook = make_test_display(display)

    env = EgoGridWorld(2)
    policy = UniformRandomPolicy(env.actions)

    env.sample_episode(policy, horizon=10, hook=display_hook)


def test_cts_grid_world(display=False):
    dummy_map = np.array([
        ['x', 'x', 'x', 'x'],
        ['x', ' ', 'G', 'x'],
        ['x', ' ', 'P', 'x'],
        ['x', 'S', 'O', 'x'],
        ['x', 'x', 'x', 'x']])

    dummy_world = ContinuousGridWorld(dummy_map, speed=1.0, noise_std=0)
    assert dummy_world.current_position == (3.0, 1.0)

    assert not dummy_world.in_pit_state()
    assert not dummy_world.in_puddle_state()
    assert not dummy_world.in_terminal_state()

    dummy_world.step('NORTH')
    dummy_world.step('NORTH')
    dummy_world.step('NORTH')

    assert not dummy_world.in_pit_state()
    assert not dummy_world.in_puddle_state()
    assert not dummy_world.in_terminal_state()
    assert dummy_world.current_position == (1.0, 1.0)

    dummy_world.step('EAST')
    assert dummy_world.in_terminal_state()
    assert not dummy_world.in_pit_state()
    assert not dummy_world.in_puddle_state()
    assert dummy_world.current_position == (1.0, 2.0)

    dummy_world.step('SOUTH')
    assert not dummy_world.in_terminal_state()
    assert not dummy_world.in_pit_state()
    assert dummy_world.in_puddle_state()
    assert dummy_world.current_position == (2.0, 2.0)

    dummy_world.step('EAST')
    assert not dummy_world.in_terminal_state()
    assert not dummy_world.in_pit_state()
    assert dummy_world.in_puddle_state()
    assert dummy_world.current_position == (2.0, 2.0)

    dummy_world.step('SOUTH')
    assert not dummy_world.in_terminal_state()
    assert dummy_world.in_pit_state()
    assert not dummy_world.in_puddle_state()
    assert dummy_world.current_position == (3.0, 2.0)

    dummy_world.step('EAST')
    assert not dummy_world.in_terminal_state()
    assert not dummy_world.in_pit_state()
    assert not dummy_world.in_puddle_state()
    assert dummy_world.current_position == (3.0, 1.0)
