import numpy as np
from collections import defaultdict

from spectral_dagger.mdp import ValueIteration
from spectral_dagger.mdp import LinearGibbsPolicy
from spectral_dagger.envs import ContinuousGridWorld
from spectral_dagger.function_approximation import RectangularTileCoding
from spectral_dagger.function_approximation import StateActionFeatureExtractor

from spectral_dagger.utils.math import geometric_sequence

from sklearn.neighbors import KernelDensity

n_expert_trajectories = 5
n_samples_per_gradient = 10
horizon = 15
smoothing = 1.0
threshold = 0.001
lmbda = 0.5
delta = 0.1
gamma = 0.99
learning_rate = geometric_sequence(0.2, tau=33)

world_map = np.array([
    ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x'],
    ['x', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x'],
    ['x', ' ', 'x', ' ', 'x', ' ', ' ', ' ', 'x'],
    ['x', ' ', 'x', 'G', 'x', ' ', 'x', ' ', 'x'],
    ['x', ' ', 'P', ' ', ' ', ' ', ' ', ' ', 'x'],
    ['x', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x'],
    ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x']])

world_map = np.array([
    ['x', 'x', 'x', 'x', 'x', 'x', 'x'],
    ['x', ' ', 'x', 'x', 'x', ' ', 'x'],
    ['x', ' ', ' ', ' ', ' ', ' ', 'x'],
    ['x', ' ', 'x', 'x', ' ', ' ', 'x'],
    ['x', ' ', 'x', 'x', ' ', ' ', 'x'],
    ['x', ' ', 'x', 'x', 'P', ' ', 'x'],
    ['x', ' ', 'x', 'x', 'P', ' ', 'x'],
    ['x', ' ', 'x', 'x', 'P', ' ', 'x'],
    ['x', ' ', 'x', 'x', 'P', ' ', 'x'],
    ['x', ' ', ' ', 'G', ' ', ' ', 'x'],
    ['x', ' ', 'x', 'x', 'x', ' ', 'x'],
    ['x', ' ', ' ', ' ', ' ', ' ', 'x'],
    ['x', ' ', 'P', 'P', 'x', ' ', 'x'],
    ['x', ' ', 'x', 'x', 'x', ' ', 'x'],
    ['x', ' ', 'x', 'x', 'x', ' ', 'x'],
    ['x', ' ', 'x', 'x', 'P', ' ', 'x'],
    ['x', ' ', 'x', 'x', 'P', ' ', 'x'],
    ['x', ' ', ' ', ' ', ' ', ' ', 'x'],
    ['x', ' ', 'x', 'x', 'x', ' ', 'x'],
    ['x', ' ', ' ', ' ', ' ', ' ', 'x'],
    ['x', 'P', 'P', 'P', 'x', ' ', 'x'],
    ['x', 'x', 'x', 'x', 'x', 'x', 'x']])
mdp = ContinuousGridWorld(
    world_map, gamma=gamma,
    rewards={'goal': 0, 'default': -1, 'puddle': -5},
    terminate_on_goal=False)

state_feature_extractor = RectangularTileCoding(
    n_tilings=1, extent=mdp.world_map.bounds.s, tile_dims=1, intercept=False)
feature_extractor = StateActionFeatureExtractor(
    state_feature_extractor, mdp.n_actions)
n_features = feature_extractor.n_features

# define expert
expert = ValueIteration().fit(mdp)

# sample expert trajectories
expert_action_data = defaultdict(list)

expert_step_wise_states = [[] for i in range(horizon)]
expert_examples = []

expert_trajectories = []
for n in range(n_expert_trajectories):
    trajectory = mdp.sample_trajectory(
        expert, horizon=horizon, display=False)
    expert_trajectories.append(trajectory)

    for t, (s, a, r) in enumerate(trajectory):
        expert_examples.append((s, a))
        expert_step_wise_states[t].append(s)

expert_step_wise_dists = []
for d in expert_step_wise_states:
    kde = KernelDensity(
        bandwidth=0.04, metric='haversine',
        kernel='gaussian', algorithm='ball_tree')
    kde.fit(d)
    expert_step_wise_dists.append(d)

# Now perform the optimization
policy = LinearGibbsPolicy(
    mdp.actions, feature_extractor,
    np.random.random(feature_extractor.n_features) - 0.5)

alpha = learning_rate.next()
while alpha > threshold:
    feature_gradient = np.zeros(n_features)
    if lmbda > 0:
        sample_trajectories = [
            mdp.sample_trajectory(
                policy, horizon=horizon, display=False)
            for k in range(n_samples_per_gradient)]

        # Calculate reward sequence for each trajectory
        rewards = []
        for tau in sample_trajectories:
            rewards.append([
                d.score([s])
                for d, (s, _, _) in zip(expert_step_wise_dists, tau)])

        # Calculate average reward for each time step
        average_reward = []
        for t in range(horizon):
            average_reward.append(np.mean([r[t] for r in rewards]))

        # Calculate gradient estimate
        for tau, tau_rewards in zip(sample_trajectories, rewards):
            adjusted_reward = tau_rewards - average_reward
            reward_weights = sum(adjusted_reward) - np.cumsum(adjusted_reward)

            gradient = np.zeros(n_features)
            for r, (s, a, _) in zip(reward_weights, trajectory):
                gradient += policy.gradient_log(s, a) * r

            feature_gradient += gradient

        norm = np.linalg.norm(feature_gradient)
        if norm > 0:
            feature_gradient /= norm

    classification_gradient = np.zeros(n_features)
    if lmbda < 1:
        # Now calculate gradient of classifier.
        for (s, a) in expert_examples:
            # Follow gradient of likelihood of the expert actions?
            classification_gradient += policy.gradient_log(s, a)

        norm = np.linalg.norm(classification_gradient)
        if norm > 0:
            classification_gradient /= norm

    policy.theta += alpha * (
        (1-lmbda) * classification_gradient + lmbda * feature_gradient)

    alpha = learning_rate.next()
    j += 1

print "Gradient descent completed after %d iterations." % j
tau = mdp.sample_trajectory(
    policy, horizon=horizon, display=True)
