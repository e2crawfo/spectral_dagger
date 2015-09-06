import numpy as np
from collections import defaultdict

from spectral_dagger.mdp import UniformRandomPolicy, ValueIteration
from spectral_dagger.mdp import LinearRewardMDP
from spectral_dagger.envs import GridWorld
from spectral_dagger.function_approximation import RectangularTileCoding
from spectral_dagger.function_approximation import StateActionFeatureExtractor
from spectral_dagger.function_approximation import discounted_features

from spectral_dagger.utils.math import laplace_smoothing, geometric_sequence

n_expert_trajectories = 5
n_sample_trajectories = 10
horizon = 15
smoothing = 1.0
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
mdp = GridWorld(
    world_map, gamma=gamma,
    rewards={'goal': 0, 'default': -1, 'puddle': -5},
    terminate_on_goal=False)

state_feature_extractor = RectangularTileCoding(
    n_tilings=1, bounds=mdp.world_map.bounds.s, granularity=1, intercept=False)
feature_extractor = StateActionFeatureExtractor(
    state_feature_extractor, mdp.n_actions)

feature_range = np.ones(feature_extractor.n_features)

# define expert
expert = ValueIteration().fit(mdp)

# sample expert trajectories
expert_action_data = defaultdict(list)

expert_trajectories = []
for n in range(n_expert_trajectories):
    trajectory = mdp.sample_trajectory(
        expert, horizon=horizon, display=False)
    expert_trajectories.append(trajectory)

    for s, a, r in trajectory:
        expert_action_data[s].append(a)

# estimate expert action distributions conditioned on state
expert_action_dist = {
    s: laplace_smoothing(smoothing, mdp.actions, expert_action_data[s])
    for s in mdp.states}

# find discounted average features for each expert trajectory
expert_features = np.mean(
    np.array([
        discounted_features(t, feature_extractor, gamma)
        for t in expert_trajectories]),
    axis=0)

# print np.where(expert_features != 0)
# print expert_features[np.where(expert_features != 0)]

sampling_policy = UniformRandomPolicy(mdp)

# obtain sampling trajectories
sample_trajectories = []

for n in range(n_sample_trajectories):
    trajectory = mdp.sample_trajectory(
        sampling_policy, horizon=horizon, display=False)
    sample_trajectories.append(trajectory)

# compute quantities required in the estimation of the gradient
pi = []
u = []
f = []

for tau in sample_trajectories:
    pi_tau = 1.0
    u_tau = 1.0

    for s, a, r in tau:
        pi_tau *= sampling_policy.action_distribution(s)[a]
        u_tau *= expert_action_dist[s][a]

    f_tau = discounted_features(tau, feature_extractor, gamma)
    # print np.where(f_tau != 0)
    # print f_tau[np.where(f_tau != 0)]

    pi.append(pi_tau)
    u.append(u_tau)
    f.append(f_tau)

# calculate epsilon
epsilon = (
    np.sqrt(-0.5 * np.log(1 - delta) / n)
    * (gamma**(horizon+1) - 1) / (gamma - 1)
    * feature_range)

n_restarts = 10
n_tests = 10

thetas = []

for i in range(n_restarts):
    learning_rate = geometric_sequence(0.2, tau=33)
    if 1:
        theta = np.random.random(feature_extractor.n_features) - 0.5
    else:
        theta = expert_features[:]

    # perform the optimization
    # linear_mdp = LinearRewardMDP(mdp, feature_extractor, theta)
    # theta_policy = ValueIteration().fit(linear_mdp)
    # mdp.sample_trajectory(theta_policy, horizon=horizon, display=False)

    n_iterations = 100
    for i in range(n_iterations):
        numerator = 0
        denominator = 0

        for pi_tau, u_tau, f_tau, tau in zip(pi, u, f, sample_trajectories):
            numerator += (u_tau / pi_tau) * np.exp(theta.dot(f_tau)) * f_tau
            denominator += (u_tau / pi_tau) * np.exp(theta.dot(f_tau))

        #assert not np.isclose(denominator, 0)

        alpha = np.sign(theta)

        theta += learning_rate.next() * (
            expert_features - numerator / denominator - alpha * epsilon)

    linear_mdp = LinearRewardMDP(mdp, feature_extractor, theta)
    theta_policy = ValueIteration().fit(linear_mdp)
    test_results = []

    for j in range(n_tests):
        tau = mdp.sample_trajectory(
            theta_policy, horizon=horizon, display=False)
        test_results.append(sum([t[2] for t in tau]))

    thetas.append((theta, np.mean(test_results)))

print [o[1] for o in thetas]
theta = max(thetas, key=lambda o: o[1])[0]
linear_mdp = LinearRewardMDP(mdp, feature_extractor, theta)
theta_policy = ValueIteration().fit(linear_mdp)
tau = mdp.sample_trajectory(theta_policy, horizon=horizon, display=True)
