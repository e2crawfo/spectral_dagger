import numpy as np

from spectral_dagger.utils import geometric_sequence
from spectral_dagger.utils import LinearGibbsPolicy

from spectral_dagger.cts_grid_world import ContinuousGridWorld
from spectral_dagger.function_approximation import RectangularTileCoding
from spectral_dagger.function_approximation import StateActionFeatureExtractor

# Simplest form of the policy gradient: sample a bunch of trajectories
# using current policy, evaluate (gradient of policy/policy) * value
# of objective function

n_samples_per_step = 5
n_steps = 100
horizon = 10
alpha = geometric_sequence(0.2, tau=33)

world_map = np.array([
    ['x', 'x', 'x', 'x', 'x'],
    ['x', 'S', ' ', ' ', 'x'],
    ['x', ' ', ' ', ' ', 'x'],
    ['x', ' ', ' ', 'G', 'x'],
    ['x', 'x', 'x', 'x', 'x']])

mdp = ContinuousGridWorld(
    world_map, speed=0.5,
    rewards={'goal': 0, 'default': -1, 'puddle': -10})

state_feature_extractor = RectangularTileCoding(
    n_tilings=1, extent=mdp.world_map.bounds.s, tile_dims=0.3)
feature_extractor = StateActionFeatureExtractor(
    state_feature_extractor, mdp.n_actions)

policy = LinearGibbsPolicy(
    mdp.actions, feature_extractor, np.zeros(feature_extractor.n_features))

R = []
average_reward = 0

display = False
for i in range(n_steps):

    gradients = []
    for j in range(n_samples_per_step):
        trajectory = mdp.sample_trajectory(
            policy, horizon=horizon, display=display)
        rewards = [t[2] for t in trajectory]

        gradient = np.zeros(feature_extractor.n_features)
        for k, (s, a, r) in enumerate(trajectory):
            feature_vectors = np.array([
                policy.feature_extractor.as_vector(s, b)
                for b in mdp.actions])

            gradient += (sum(rewards[k:]) - average_reward) * (
                feature_extractor.as_vector(s, a)
                - feature_vectors.T.dot(policy.action_distribution(s)))

        total_reward = sum(rewards)

        gradients.append(gradient)

        count = float(i * j + 1)
        average_reward = ((count-1) * average_reward + total_reward) / count
        print average_reward
        R.append(total_reward)

    policy.theta += alpha.next() * np.mean(gradients, axis=0)

import matplotlib.pyplot as plt

plt.plot(R)
plt.gca().set_ylim((min(R)-10, max(R) + 10))
plt.show()