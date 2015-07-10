import numpy as np

from spectral_dagger.mdp import MDPPolicy
from spectral_dagger.td import LinearGradientTD
from spectral_dagger.utils import geometric_sequence

from spectral_dagger.cts_grid_world import ContinuousGridWorld
from spectral_dagger.function_approximation import RectangularTileCoding
from spectral_dagger.function_approximation import FeatureExtractor

# Dummy form of the policy gradient: sample a bunch of trajectories
# using current policy, evaluate (gradient of policy/policy) * value
# of objective function

class SoftmaxPolicy(MDPPolicy):
    def __init__(self, mdp, feature_extractor, phi):
        self.mdp = mdp
        self.feature_extractor = feature_extractor
        self.phi = phi.flatten()

        assert len(self.phi == self.feature_extractor.n_features)

    def reset(self, state):
        self.current_state = state

    def update(self, action, state, reward=None):
        self.current_state = state

    def get_action(self):
        probs = self.action_distribution(self.current_state)
        sample = np.random.multinomial(1, probs)
        action = np.where(sample > 0)[0][0]

        return action

    def action_distribution(self, state):
        feature_vectors = np.array([
            self.feature_extractor.as_vector(state, a)
            for a in self.mdp.actions])
        probs = np.exp(feature_vectors.dot(self.phi))
        probs = probs / sum(probs)

        return probs


n_samples_per_step = 5
n_steps = 500
horizon = 20
alpha = geometric_sequence(0.2, tau=166)

world_map = np.array([
    ['x', 'x', 'x', 'x', 'x'],
    ['x', ' ', ' ', ' ', 'x'],
    ['x', ' ', ' ', ' ', 'x'],
    ['x', ' ', ' ', 'G', 'x'],
    ['x', 'x', 'x', 'x', 'x']])

mdp = ContinuousGridWorld(
        world_map, speed=0.5, rewards={'goal': 0, 'default': -1, 'puddle':-10})

class StateActionFeatureExtractor(FeatureExtractor):
    def __init__(self, state_feature_extractor, n_actions):
        self.state_feature_extractor = state_feature_extractor
        self.n_actions = n_actions
        self.n_state_features = self.state_feature_extractor.n_features
        self._n_features = self.n_actions * self.n_state_features

    def as_vector(self, state, action):
        state_rep = self.state_feature_extractor.as_vector(state)

        vector = np.zeros(self.n_features)
        action = int(action)
        lo = action * self.n_state_features
        hi = lo + self.n_state_features
        vector[lo:hi] = state_rep

        return vector


state_feature_extractor = RectangularTileCoding(
    n_tilings=2, bounds=mdp.world_map.bounds.s, granularity=0.3)
feature_extractor = StateActionFeatureExtractor(
    state_feature_extractor, mdp.n_actions)

policy = SoftmaxPolicy(
    mdp, feature_extractor, np.random.random(feature_extractor.n_features))

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

        # gradients.append(gradient * total_reward)
        gradients.append(gradient)

        count = float(i * j + 1)
        average_reward = ((count-1) * average_reward + total_reward) / count
        R.append(total_reward)

    policy.phi += alpha.next() * np.mean(np.array(gradients).T, axis=1)

import matplotlib.pyplot as plt

plt.plot(R)
plt.gca().set_ylim((min(R)-10, max(R) + 10))
plt.show()