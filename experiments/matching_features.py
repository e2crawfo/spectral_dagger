import numpy as np

from spectral_dagger.utils import geometric_sequence
from spectral_dagger.utils import LinearGibbsPolicy
from spectral_dagger.value_iteration import ValueIteration
from spectral_dagger.grid_world import GridWorld
from spectral_dagger.function_approximation import RectangularTileCoding
from spectral_dagger.function_approximation import StateActionFeatureExtractor
from spectral_dagger.function_approximation import discounted_features


"""
Starting with a simple version. No importance sampling, just policy gradients
on the feature matching, with the classification error on all encountered
data sets as regularization.
"""

n_expert_trajectories = 5
horizon = 15
gamma = 0.99
n_iters = 2
n_samples_per_iter = 5
n_samples_per_gradient = 10
threshold = 0.001
lmbda = 0.01

world_map = np.array([
    ['x', 'x', 'x', 'x', 'x', 'x', 'x'],
    ['x', ' ', ' ', ' ', ' ', ' ', 'x'],
    ['x', ' ', ' ', ' ', ' ', ' ', 'x'],
    ['x', ' ', ' ', ' ', ' ', ' ', 'x'],
    ['x', ' ', ' ', ' ', ' ', ' ', 'x'],
    ['x', ' ', 'T', ' ', 'T', ' ', 'x'],
    ['x', 'x', 'T', ' ', 'T', 'x', 'x'],
    ['x', ' ', ' ', ' ', ' ', ' ', 'x'],
    ['x', ' ', ' ', 'G', ' ', ' ', 'x'],
    ['x', 'x', 'x', 'x', 'x', 'x', 'x']])

mdp = GridWorld(
    world_map, gamma=gamma,
    rewards={'goal': 0, 'default': -1, 'puddle': -5, 'trap': -1},
    terminate=False)
state_feature_extractor = RectangularTileCoding(
    n_tilings=1, bounds=mdp.world_map.bounds.s, granularity=1, intercept=False)
feature_extractor = StateActionFeatureExtractor(
    state_feature_extractor, mdp.n_actions)

n_features = feature_extractor.n_features

# define expert
expert = ValueIteration().fit(mdp)

expert_examples = []

# gather expert trajectories
expert_trajectories = []
for n in range(n_expert_trajectories):
    trajectory = mdp.sample_trajectory(
        expert, horizon=horizon, display=False)
    expert_trajectories.append(trajectory)

    for s, a, r in trajectory:
        expert_examples.append((s, a))

# calculate discounted expert features
expert_features = [
    discounted_features(t, feature_extractor, gamma)
    for t in expert_trajectories]

expert_features = np.mean(expert_features, axis=0)

policies = []

for i in range(n_iters):
    average_reward = 0

    policy = LinearGibbsPolicy(
        mdp, feature_extractor,
        np.random.random(feature_extractor.n_features) - 0.5)

    learning_rate = geometric_sequence(0.2, tau=33)

    j = 0

    alpha = learning_rate.next()
    while alpha > threshold:

        feature_gradient = np.zeros(n_features)
        if lmbda > 0:
            # Sample trajectories at current theta
            for k in range(n_samples_per_gradient):
                trajectory = mdp.sample_trajectory(
                    policy, horizon=horizon, display=False)

                gradient = np.zeros(n_features)

                for k, (s, a, r) in enumerate(trajectory):
                    feature_vectors = np.array([
                        feature_extractor.as_vector(s, b)
                        for b in mdp.actions])

                    gradient += (
                        feature_extractor.as_vector(s, a)
                        - feature_vectors.T.dot(policy.action_distribution(s)))

                df = discounted_features(trajectory, feature_extractor, gamma)
                reward = df.dot(df - 2 * expert_features)
                print "Reward: ", reward

                gradient *= reward - average_reward

                count = float(j * k + 1)
                average_reward = ((count-1) * average_reward + reward) / count

                feature_gradient += -gradient

            feature_gradient /= n_samples_per_gradient

        classification_gradient = np.zeros(n_features)
        if lmbda < 1:
            # Now calculate gradient of classifier.
            # Recall that we're using linear gibbs for now.
            for (s, a) in expert_examples:
                feature_vectors = np.array([
                    policy.feature_extractor.as_vector(s, b)
                    for b in mdp.actions])

                classification_gradient += (
                    feature_extractor.as_vector(s, a)
                    - feature_vectors.T.dot(policy.action_distribution(s)))

        policy.phi += learning_rate.next() * (
            (1-lmbda) * classification_gradient + lmbda * feature_gradient)

        alpha = learning_rate.next()
        j += 1

    print "Gradient descent completed after %d iterations." % j

    # Now we're done the iteration. Get expert demonstrations.
    policies.append(policy)

    if lmbda < 1:
        for j in range(n_samples_per_iter):
            tau = mdp.sample_trajectory(
                policy, horizon=horizon, display=False)
            expert_examples.extend([(s, a) for s, a, _ in tau])

    print "After iteration %d" % i
    tau = mdp.sample_trajectory(
        policy, horizon=horizon, display=True)

