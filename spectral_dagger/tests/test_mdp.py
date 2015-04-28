from spectral_dagger.policy_iteration import PolicyIteration
from spectral_dagger.value_iteration import ValueIteration

import grid_world


def test_policy_iteration():

    env = grid_world.GridWorld(gamma=0.9)

    alg = PolicyIteration(threshold=0.0001)
    policy = alg.fit(env)

    trajectory = env.sample_trajectory(
        policy, horizon=20, reset=True, display=True)

    print trajectory


def test_value_iteration():

    env = grid_world.GridWorld(gamma=0.9)

    alg = ValueIteration(threshold=0.0001)
    policy = alg.fit(env)

    trajectory = env.sample_trajectory(
        policy, horizon=20, reset=True, display=True)

    print trajectory
