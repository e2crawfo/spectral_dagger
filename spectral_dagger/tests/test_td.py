import numpy as np

from spectral_dagger.td import QLearning, Sarsa, TD
from spectral_dagger.grid_world import GridWorld
from spectral_dagger.random_walk import RandomWalk

from spectral_dagger.cts_grid_world import ContinuousGridWorld
from spectral_dagger.utils import SingleActionMDP
from spectral_dagger.mdp import UniformRandomPolicy
from spectral_dagger.function_approximation import RectangularTileCoding
from spectral_dagger.function_approximation import CircularCoarseCoding
from spectral_dagger.td import LinearGradientTD, LinearGradientSarsa
from spectral_dagger.utils import geometric_sequence


cliff_world = np.array([
    ['x', 'x', 'x', 'x', 'x', 'x'],
    ['x', 'S', ' ', ' ', 'O', 'x'],
    ['x', ' ', ' ', ' ', 'O', 'x'],
    ['x', ' ', ' ', ' ', 'O', 'x'],
    ['x', ' ', ' ', ' ', 'G', 'x'],
    ['x', 'x', 'x', 'x', 'x', 'x']]
)

canyon_world = np.array([
    ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x'],
    ['x', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x'],
    ['x', ' ', ' ', 'O', 'O', 'O', ' ', ' ', 'x'],
    ['x', 'S', ' ', ' ', ' ', 'O', 'G', ' ', 'x'],
    ['x', ' ', ' ', 'O', 'O', 'O', ' ', ' ', 'x'],
    ['x', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x'],
    ['x', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x'],
    ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x']]
)
noise = 0.1
gamma = 0.9
epsilon = 0.1
alpha = [0.1, 0.2]
lmbda = [0, 0.3, 0.4, 1]
n_episodes = 100


def pytest_generate_tests(metafunc):
    if "alpha" in metafunc.funcargnames:
        metafunc.parametrize("alpha", alpha)

    if "lmbda" in metafunc.funcargnames:
        metafunc.parametrize("lmbda", lmbda)


def test_td_prediction(alpha, lmbda, display=False):
    env = RandomWalk(10, p=0.5, gamma=1.0)

    policy = TD(
        env, alpha=alpha, L=lmbda,
        V_0=np.random.random(env.n_states))

    env.sample_trajectory(
        policy, reset=True, display=display)
    env.sample_trajectory(
        policy, reset=True, display=display)


def test_linear_gtd(display=False):
    dummy_map = np.array([
        ['x', 'x', 'x', 'x'],
        ['x', ' ', 'G', 'x'],
        ['x', ' ', ' ', 'x'],
        ['x', ' ', ' ', 'x'],
        ['x', 'S', 'P', 'x'],
        ['x', 'x', 'x', 'x']])

    dummy_world = ContinuousGridWorld(
        dummy_map, speed=0.3, rewards={'puddle': -10})

    random_world = SingleActionMDP(
        dummy_world, UniformRandomPolicy(dummy_world))

    feature_extractor = RectangularTileCoding(
        n_tilings=3, bounds=dummy_world.world_map.bounds.s, granularity=0.5)
    linear_gtd = LinearGradientTD(
        random_world, feature_extractor, geometric_sequence(0.2, 20))

    n_episodes = 10
    for i in range(n_episodes):
        random_world.sample_trajectory(
            policy=linear_gtd, display=display, reset=True)


def test_q_learning(alpha, display=False):
    env = GridWorld(cliff_world, noise=noise, gamma=gamma)

    policy = QLearning(
        env, alpha=alpha, epsilon=epsilon,
        Q_0=np.random.random((env.n_states, env.n_actions)))

    env.sample_trajectory(
        policy, reset=True, display=display)
    env.sample_trajectory(
        policy, reset=True, display=display)


def test_sarsa(alpha, lmbda, display=False):
    env = GridWorld(cliff_world, noise=noise, gamma=gamma)

    policy = Sarsa(
        env, alpha=alpha, L=lmbda, epsilon=epsilon,
        Q_0=np.random.random((env.n_states, env.n_actions)))

    env.sample_trajectory(
        policy, reset=True, display=display)
    env.sample_trajectory(
        policy, reset=True, display=display)


def test_linear_sarsa_gtd(display=False):
    dummy_map = np.array([
        ['x', 'x', 'x', 'x'],
        ['x', ' ', 'G', 'x'],
        ['x', ' ', ' ', 'x'],
        ['x', ' ', ' ', 'x'],
        ['x', 'S', 'P', 'x'],
        ['x', 'x', 'x', 'x']])

    dummy_world = ContinuousGridWorld(
        dummy_map, speed=0.3, rewards={'puddle': -10})

    env_bounds = dummy_world.world_map.bounds.s
    feature_extractor = CircularCoarseCoding(
        n_circles=100, bounds=env_bounds, radius=0.5)
    linear_gsarsa = LinearGradientSarsa(
        dummy_world, feature_extractor, geometric_sequence(0.2, 20))

    n_episodes = 10
    for i in range(n_episodes):
        dummy_world.sample_trajectory(
            policy=linear_gsarsa, display=display, reset=True)
