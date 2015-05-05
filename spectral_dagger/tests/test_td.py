import numpy as np
import matplotlib.pyplot as plt

from spectral_dagger.td import QLearning, Sarsa
from spectral_dagger.grid_world import GridWorld

cliff_world = np.array([
    ['x', 'x', 'x', 'x', 'x', 'x'],
    ['x', 'A', ' ', ' ', 'O', 'x'],
    ['x', ' ', ' ', ' ', 'O', 'x'],
    ['x', ' ', ' ', ' ', 'O', 'x'],
    ['x', ' ', ' ', ' ', 'G', 'x'],
    ['x', 'x', 'x', 'x', 'x', 'x']]
)

canyon_world = np.array([
    ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x'],
    ['x', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x'],
    ['x', ' ', ' ', 'O', 'O', 'O', ' ', ' ', 'x'],
    ['x', 'A', ' ', ' ', ' ', 'O', 'G', ' ', 'x'],
    ['x', ' ', ' ', 'O', 'O', 'O', ' ', ' ', 'x'],
    ['x', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x'],
    ['x', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x'],
    ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x']]
)
noise = 0.1
gamma = 0.9
epsilon = 0.1
alpha = 0.01
n_episodes = 10000


def pytest_generate_tests(metafunc):
    if "algorithm" in metafunc.funcargnames:
        metafunc.parametrize(
            "algorithm", [QLearning, Sarsa])


def test_td(algorithm):
    env = GridWorld(cliff_world, noise=noise, gamma=gamma)

    policy = algorithm(
        env, alpha=alpha, epsilon=epsilon,
        Q_0=np.random.random((env.num_states, env.num_actions)))

    env.sample_trajectory(
        policy, reset=True, display=False)
    env.sample_trajectory(
        policy, reset=True, display=False)


def plot_average_reward(env, policy, n_episodes):
    rewards = []
    for i in range(n_episodes):
        trajectory = env.sample_trajectory(
            policy, reset=True, display=False, return_reward=True)
        rewards.append([t[2] for t in trajectory])

    plt.plot(range(n_episodes), [np.mean(r) for r in rewards])
    plt.show()
