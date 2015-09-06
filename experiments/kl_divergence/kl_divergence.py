import numpy as np
from collections import defaultdict

from spectral_dagger.utils.math import geometric_sequence, laplace_smoothing
from spectral_dagger.mdp import LinearGibbsPolicy
from spectral_dagger.mdp import ValueIteration
from spectral_dagger.envs import GridWorld
from spectral_dagger.function_approximation import RectangularTileCoding
from spectral_dagger.function_approximation import StateActionFeatureExtractor


"""
Starting with a simple version. No importance sampling, just policy gradients
on the feature matching, with the classification error on all encountered
data sets as regularization.
"""


def f(
        n_expert_trajectories=5, horizon=10, gamma=0.99, n_restarts=1,
        n_samples_per_gradient=10, threshold=0.001, smoothing=1.0,
        n_extra_actions=4, temperature=1.0, init_with_classification=True):

    world_map = np.array([
        ['x', 'x', 'x', 'x', 'x', 'x', 'x'],
        ['x', 'S', ' ', ' ', ' ', ' ', 'x'],
        ['x', ' ', ' ', ' ', ' ', ' ', 'x'],
        ['x', ' ', ' ', ' ', 'G', ' ', 'x'],
        ['x', ' ', ' ', ' ', ' ', ' ', 'x'],
        ['x', 'x', 'x', 'x', 'x', 'x', 'x']])

    mdp = GridWorld(
        world_map, gamma=gamma,
        rewards={'goal': 0, 'default': -1, 'puddle': -5, 'trap': -1},
        terminate_on_goal=False)
    state_feature_extractor = RectangularTileCoding(
        n_tilings=1, bounds=mdp.world_map.bounds.s,
        granularity=1, intercept=True)

    feature_extractor = StateActionFeatureExtractor(
        state_feature_extractor, mdp.n_actions)

    n_features = feature_extractor.n_features

    # define expert
    expert = ValueIteration().fit(mdp)

    expert_action_data = defaultdict(list)
    expert_examples = []

    # gather expert trajectories
    expert_trajectories = []
    for n in range(n_expert_trajectories):
        trajectory = mdp.sample_trajectory(
            expert, horizon=horizon, display=False)
        expert_trajectories.append(trajectory)

        for s, a, r in trajectory:
            expert_action_data[s].append(a)
            expert_examples.append((s, a))

    padded_actions = (
        mdp.actions + range(mdp.n_actions, mdp.n_actions + n_extra_actions))
    expert_action_dist = {
        s: laplace_smoothing(smoothing, padded_actions, expert_action_data[s])
        for s in mdp.states}

    policies = []

    for i in range(n_restarts):
        average_reward = 0

        policy = LinearGibbsPolicy(
            mdp, feature_extractor,
            np.random.random(feature_extractor.n_features) - 0.5)

        # Initialize weights by doing classification
        if init_with_classification:
            learning_rate = geometric_sequence(0.2, tau=33)
            alpha = learning_rate.next()

            while alpha > threshold:
                classification_gradient = np.zeros(n_features)

                # Now calculate gradient of classifier.
                # Recall that we're using linear gibbs for now.
                for (s, a) in expert_examples:
                    error = (
                        policy.action_distribution(s)[a]
                        - expert_action_dist[s][a])

                    classification_gradient += error * policy.gradient(s, a)

                norm = np.linalg.norm(classification_gradient)
                if norm > 0:
                    classification_gradient /= norm

                # Move in direction of anti-gradient since we're minimizing
                policy.phi += -alpha * classification_gradient
                alpha = learning_rate.next()

            trajectory = mdp.sample_trajectory(
                policy, horizon=horizon, display=True)

        learning_rate = geometric_sequence(0.2, tau=33)
        j = 0

        alpha = learning_rate.next()
        while alpha > threshold:

            gradient = np.zeros(n_features)

            print "Iter: ", j

            # Sample trajectories at current theta
            for k in range(n_samples_per_gradient):
                trajectory = mdp.sample_trajectory(
                    policy, horizon=horizon, display=False)

                sample_gradient = np.zeros(n_features)
                et = np.zeros(n_features)

                cumulative_reward = 0
                for l, (s, a, _) in enumerate(trajectory):
                    et += policy.gradient_log(s, a)

                    reward = np.log(
                        policy.action_distribution(s)[a]
                        / expert_action_dist[s][a]) - average_reward

                    cumulative_reward += reward
                    sample_gradient += reward * et

                    count = float(
                        j * n_samples_per_gradient * horizon
                        + k * horizon + l + 1)
                    average_reward = (
                        ((count-1) * average_reward + reward) / count)

                gradient += sample_gradient

            print "Average Reward: ", average_reward
            norm = np.linalg.norm(gradient)
            if norm > 0:
                gradient /= norm

            # Move in direction of anti-gradient since we're minimizing
            policy.phi += -alpha * gradient

            alpha = learning_rate.next()
            j += 1

        print "Gradient descent completed after %d iterations." % j
        print policies.append((policy, average_reward))

        print "After iteration %d" % i
        mdp.sample_trajectory(
            policy, horizon=horizon, display=True)

    print [o[1] for o in policies]
    theta_policy = min(policies, key=lambda o: o[1])[0]
    mdp.sample_trajectory(theta_policy, horizon=horizon, display=True)

if __name__ == "__main__":
    f()
