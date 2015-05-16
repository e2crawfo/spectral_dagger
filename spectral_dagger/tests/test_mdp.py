from spectral_dagger.policy_iteration import PolicyIteration
from spectral_dagger.value_iteration import ValueIteration

import spectral_dagger.grid_world as grid_world

import numpy as np


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
