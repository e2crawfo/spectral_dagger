import matplotlib.pyplot as plt
import numpy as np

from td import Sarsa, TD, LinearGradientSarsa
from grid_world import GridWorld
from .mdp import evaluate_policy
from .utils import geometric_sequence
from cts_grid_world import ContinuousGridWorld
from .function_approximation import RectangularTileCoding


# LinearGradientSarsa ************************

def run_td_gsarsa(
        name, env_class, env_kwargs,
        policy_class, policy_kwargs,
        fe_class, fe_kwargs, n_episodes, n_trials, horizon):

    env = env_class(**env_kwargs)

    alpha = policy_kwargs['alpha']
    epsilon = policy_kwargs['epsilon']

    policy_kwargs['alpha'] = geometric_sequence(
        policy_kwargs['alpha'], tau=100)
    policy_kwargs['epsilon'] = geometric_sequence(
        policy_kwargs['epsilon'], tau=100)
    policy_kwargs['feature_extractor'] = fe_class(**fe_kwargs)

    rewards = []
    max_theta = -np.inf
    min_theta = -np.inf

    for j in range(n_trials):
        policy = policy_class(env, **policy_kwargs)

        episodes = env.sample_episodes(n_episodes, policy, horizon)
        for ep in episodes:
            rewards.extend([r for a, s, r in ep])

        max_theta = max(max_theta, np.max(np.max(policy.theta)))
        min_theta = min(min_theta, np.min(np.min(policy.theta)))

    return {
        'average_reward': np.mean(rewards),
        'n_episodes': n_episodes,
        'n_trials': n_trials,
        'name': name,
        'epsilon': epsilon,
        'alpha': alpha,
        'L': policy_kwargs['L'],
        'min_theta': min_theta,
        'max_theta': max_theta
    }


def task_generate_gsarsa_data():
    a_0 = np.linspace(0.1, 0.3, 5)
    e_0 = np.linspace(0.1, 0.3, 5)
    L = np.linspace(0, 0.9, 5)

    horizon = 20
    n_episodes = 30
    n_trials = 10

    dummy_map = np.array([
        ['x', 'x', 'x', 'x', 'x'],
        ['x', ' ', ' ', 'G', 'x'],
        ['x', ' ', ' ', ' ', 'x'],
        ['x', ' ', ' ', ' ', 'x'],
        ['x', 'x', 'x', 'x', 'x']])
    speed = 0.5

    env_class = ContinuousGridWorld
    env_kwargs = {
        'world_map': dummy_map,
        'speed': speed,
    }

    dummy_world = ContinuousGridWorld(**env_kwargs)

    fe_kwargs = {
        'n_tilings': 1,
        'extent': dummy_world.world_map.bounds.s,
        'tile_dims': 0.5
    }

    fe_class = RectangularTileCoding

    policy_class = LinearGradientSarsa

    for e in e_0:
        for a in a_0:
            for l in L:
                policy_kwargs = {
                    'alpha': a,
                    'epsilon': e,
                    'L': l,
                }

                task_name = 'e_0: %f, a_0: %f, L: %f' % (e, a, l)
                task_args = [
                    task_name, env_class, env_kwargs,
                    policy_class, policy_kwargs,
                    fe_class, fe_kwargs,
                    n_episodes, n_trials, horizon]

                yield {
                    'name': task_name,
                    'actions': [(run_td_gsarsa, task_args)],
                }


def plot_gsarsa(output):
    fig = plt.figure()
    ax = plt.gca()
    ax.set_ylim((0, 3))
    ax.set_xlabel("alpha")
    ax.set_ylabel("average reward")

    epsilon = sorted(set(v['epsilon'] for v in list(output.values())))
    L = sorted(set(v['L'] for v in list(output.values())))

    for e in epsilon:
        for l in L:
            values = [v for v in list(output.values()) if v['epsilon'] == e and v['L'] == l]
            values = sorted(values, key=lambda v: v['alpha'])
            plt.plot(
                [v['alpha'] for v in values],
                [v['average_reward'] for v in values],
                label="epsilon: %f, L: %f" % (e, l))

    plt.legend()
    plt.show()


def task_plot_gsarsa():
    return {
        'actions': [plot_gsarsa],
        'getargs': {'output': ('generate_gsarsa_data', None)}
    }


# PREDICTION *********************************
def run_td_prediction(
        name, env_class, env_kwargs,
        policy_class, policy_kwargs, n_episodes, n_trials, true_V):

    env = env_class(**env_kwargs)

    errors = []

    for j in range(n_trials):
        policy = policy_class(env, **policy_kwargs)

        for i in range(n_episodes):
            env.sample_episode(policy)
            errors.append(np.mean((policy.V - true_V)**2))

    return {
        'rmse': np.sqrt(np.mean(errors)),
        'n_episodes': n_episodes,
        'n_trials': n_trials,
        'name': name,
        'lmbda': policy_kwargs['L'],
        'alpha': policy_kwargs['alpha']
    }


def task_generate_prediction_data():
    from random_walk import RandomWalk

    n_states = 19
    p = 0.5
    gamma = 1.0

    n_trials = 100
    n_episodes = 11

    alpha = np.linspace(0.01, 1, 10)
    lmbda = np.concatenate(
        (np.linspace(0.0, 0.9, 10), np.linspace(0.925, 1, 4)))

    env_class = RandomWalk
    env_kwargs = {
        'n_states': n_states,
        'p': p,
        'gamma': gamma
    }

    c = RandomWalk(**env_kwargs)
    true_V = evaluate_policy(c)

    print("True V: ")
    print(true_V)

    policy_class = TD

    for l in lmbda:
        for a in alpha:
            V_0 = 0.5 * np.ones(n_states)
            V_0[[0, -1]] = 0

            policy_kwargs = {
                'alpha': a,
                'L': l,
                'V_0': V_0
            }

            task_name = 'Lambda: %f, alpha: %f' % (l, a)
            task_args = [
                task_name, env_class, env_kwargs,
                policy_class, policy_kwargs, n_episodes, n_trials, true_V]

            yield {
                'name': task_name,
                'actions': [(run_td_prediction, task_args)],
            }


def plot_prediction(output):
    fig = plt.figure()
    ax = plt.gca()
    ax.set_ylim((0, 1))
    ax.set_xlabel("alpha")
    ax.set_ylabel("rmse")

    lmbda = sorted(set(v['lmbda'] for v in list(output.values())))

    for l in lmbda:
        values = [v for v in list(output.values()) if v['lmbda'] == l]
        values = sorted(values, key=lambda v: v['alpha'])
        plt.plot(
            [v['alpha'] for v in values],
            [v['rmse'] for v in values],
            label="lambda: %f" % l)

    plt.legend()
    plt.show()


def task_plot_prediction():
    return {
        'actions': [plot_prediction],
        'getargs': {'output': ('generate_prediction_data', None)}
    }


# CONTROL *********************************
def run_td_control(
        name, env_class, env_kwargs,
        policy_class, policy_kwargs, n_episodes, n_trials):

    env = env_class(**env_kwargs)

    rewards = [[] for i in range(n_episodes)]

    for j in range(n_trials):
        policy = policy_class(env, **policy_kwargs)

        episodes = env.sample_episodes(n_episodes, policy)
        for i, ep in enumerate(episodes):
            rewards[i].append(np.mean([r for a, s, r in ep]))

    return {
        'reward': [np.mean(r) for r in rewards],
        'n_episodes': n_episodes,
        'name': name,
        'lmbda': policy_kwargs['L'],
        'alpha': policy_kwargs['alpha']
    }


def task_generate_control_data():
    canyon_world = np.array([
        ['x', 'x', 'x', 'x', 'x', 'x', 'x'],
        ['x', ' ', ' ', ' ', ' ', 'G', 'x'],
        ['x', 'S', ' ', ' ', ' ', ' ', 'x'],
        ['x', 'x', 'x', 'x', 'x', 'x', 'x']]
    )

    noise = 0.0
    epsilon = 0.1
    gamma = 0.9

    n_trials = 100
    n_episodes = 11

    alpha = np.linspace(0.01, 0.3, 10)
    lmbda = np.linspace(0.0, 0.8, 10)

    env_class = GridWorld
    env_kwargs = {
        'world_map': canyon_world,
        'noise': noise,
        'gamma': gamma
    }

    policy_class = Sarsa

    for l in lmbda:
        for a in alpha:
            policy_kwargs = {
                'alpha': a,
                'L': l,
                'epsilon': epsilon,
            }

            task_name = 'Lambda: %f, alpha: %f' % (l, a)
            task_args = [
                task_name, env_class, env_kwargs,
                policy_class, policy_kwargs, n_episodes, n_trials]

            yield {
                'name': task_name,
                'actions': [(run_td_control, task_args)],
            }


def plot_control(output):
    fig = plt.figure()
    plt.gca().set_ylim((0, 6))

    lmbda = sorted(set(v['lmbda'] for v in list(output.values())))

    for l in lmbda:
        values = [v for v in list(output.values()) if v['lmbda'] == l]
        values = sorted(values, key=lambda v: v['alpha'])
        plt.plot(
            [v['alpha'] for v in values],
            [v['rmse'][10] for v in values],
            label="lambda: %f" % l)

    plt.legend()
    plt.show()


def task_plot_control():
    return {
        'actions': [plot_control],
        'getargs': {'output': ('generate_control_data', None)}
    }
