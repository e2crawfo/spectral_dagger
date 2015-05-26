import numpy as np
import matplotlib.pyplot as plt

from spectral_dagger.td import QLearning, Sarsa, TD
from spectral_dagger.grid_world import GridWorld
from spectral_dagger.random_walk import RandomWalk

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


def plot_average_reward(env, policy, n_episodes):
    rewards = []
    for i in range(n_episodes):
        trajectory = env.sample_trajectory(
            policy, reset=True, display=False, return_reward=True)
        rewards.append([t[2] for t in trajectory])

    plt.plot(range(n_episodes), [np.mean(r) for r in rewards])
    plt.show()
