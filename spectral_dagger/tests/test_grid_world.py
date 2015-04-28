from spectral_dagger.grid_world import GridWorld, EgoGridWorld
from mdp import UniformRandomPolicy as MDPUniformRandomPolicy
from pomdp import UniformRandomPolicy as POMDPUniformRandomPolicy


def test_grid_world():
    env = GridWorld()
    print str(env)

    policy = MDPUniformRandomPolicy(env)

    trajectory = env.sample_trajectory(
        policy, horizon=10, reset=True, display=True)

    print trajectory


def test_ego_grid_world():
    env = EgoGridWorld(2)
    print str(env)

    policy = POMDPUniformRandomPolicy(env)

    trajectory = env.sample_trajectory(
        policy, horizon=10, reset=True, display=True)

    print trajectory