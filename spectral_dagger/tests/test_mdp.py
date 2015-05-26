from spectral_dagger.random_walk import RandomWalk
from spectral_dagger.td import TD
from spectral_dagger.policy_iteration import PolicyIteration
from spectral_dagger.value_iteration import ValueIteration

import spectral_dagger.grid_world as grid_world


def test_simple(display=False):

    # 1 action mdp (so a markov chain), no policy
    mdp = RandomWalk(10)
    trajectory = mdp.sample_trajectory(
        reset=True, return_reward=True, display=display)

    # 1 action mdp (so a markov chain), a learning policy
    policy = TD(mdp, alpha=0.1)
    trajectory = mdp.sample_trajectory(
        policy=policy, reset=True, return_reward=True, display=display)


def test_policy_iteration(display=False):
    env = grid_world.GridWorld(gamma=0.9)

    alg = PolicyIteration(threshold=0.0001)
    policy = alg.fit(env)

    env.sample_trajectory(
        policy, horizon=20, reset=True, display=display)


def test_value_iteration(display=False):

    env = grid_world.GridWorld(gamma=0.9)

    alg = ValueIteration(threshold=0.0001)
    policy = alg.fit(env)

    env.sample_trajectory(
        policy, horizon=20, reset=True, display=display)
