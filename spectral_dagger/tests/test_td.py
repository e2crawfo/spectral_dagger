import numpy as np

from spectral_dagger import sample_episodes
from spectral_dagger.tests.conftest import make_test_display
from spectral_dagger.mdp import QLearning, Sarsa, TD
from spectral_dagger.envs import GridWorld, LinearMarkovChain

from spectral_dagger.envs import ContinuousGridWorld
from spectral_dagger.mdp import UniformRandomPolicy
from spectral_dagger.function_approximation import RectangularTileCoding
from spectral_dagger.function_approximation import CircularCoarseCoding
from spectral_dagger.mdp import LinearGradientTD, LinearGradientSarsa
from spectral_dagger.utils.math import geometric_sequence


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
    display_hook = make_test_display(display)

    env = LinearMarkovChain(10, p=0.5, gamma=1.0)

    policy = TD(
        env, alpha=alpha, L=lmbda,
        V_0=np.random.random(env.n_states))

    sample_episodes(2, env, policy, hook=display_hook)


def test_linear_gtd(display=False):
    display_hook = make_test_display(display)

    dummy_map = np.array([
        ['x', 'x', 'x', 'x'],
        ['x', ' ', 'G', 'x'],
        ['x', ' ', ' ', 'x'],
        ['x', ' ', ' ', 'x'],
        ['x', 'S', 'P', 'x'],
        ['x', 'x', 'x', 'x']])

    dummy_world = ContinuousGridWorld(
        dummy_map, speed=0.3, rewards={'puddle': -10})

    policies = [UniformRandomPolicy(dummy_world.actions)]

    feature_extractor = RectangularTileCoding(
        n_tilings=3, extent=dummy_world.world_map.bounds.s, tile_counts=5)
    linear_gtd = LinearGradientTD(
        dummy_world, feature_extractor, geometric_sequence(0.2, 20))
    policies.append(linear_gtd)

    n_episodes = 10
    sample_episodes(
        n_episodes, dummy_world, policies, horizon=10, hook=display_hook)


def test_q_learning(alpha, display=False):
    display_hook = make_test_display(display)

    env = GridWorld(cliff_world, noise=noise, gamma=gamma)

    policy = QLearning(
        env, alpha=alpha, epsilon=epsilon,
        Q_0=np.random.random((env.n_states, env.n_actions)))

    sample_episodes(2, env, policy, hook=display_hook)


def test_sarsa(alpha, lmbda, display=False):
    display_hook = make_test_display(display)

    env = GridWorld(cliff_world, noise=noise, gamma=gamma)

    policy = Sarsa(
        env, alpha=alpha, L=lmbda, epsilon=epsilon,
        Q_0=np.random.random((env.n_states, env.n_actions)))

    sample_episodes(2, env, policy, hook=display_hook)


def test_linear_sarsa_gtd(display=False):
    display_hook = make_test_display(display)

    dummy_map = np.array([
        ['x', 'x', 'x', 'x'],
        ['x', ' ', 'G', 'x'],
        ['x', ' ', ' ', 'x'],
        ['x', ' ', ' ', 'x'],
        ['x', 'S', ' ', 'x'],
        ['x', 'x', 'x', 'x']])

    dummy_world = ContinuousGridWorld(
        dummy_map, speed=0.3, rewards={'puddle': -10})

    env_bounds = dummy_world.world_map.bounds.s
    feature_extractor = CircularCoarseCoding(
        n_circles=100, bounds=env_bounds, radius=0.5)
    linear_gsarsa = LinearGradientSarsa(
        dummy_world, feature_extractor, geometric_sequence(0.2, 20))

    n_episodes = 10
    sample_episodes(
        n_episodes, dummy_world, policy=linear_gsarsa, hook=display_hook)
