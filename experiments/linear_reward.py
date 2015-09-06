import numpy as np

from spectral_dagger.mdp import ValueIteration, LinearRewardMDP
from spectral_dagger.envs import GridWorld
from spectral_dagger.function_approximation import RectangularTileCoding
from spectral_dagger.function_approximation import StateActionFeatureExtractor

"""
Take an MDP, fits its reward function using linear regression, then solve the
MDP with the fit reward function. Should give optimal polices that are similar
to optimal policies on the original MDP (if the true reward function is nearly
a linear function of the state features, that is).
"""

gamma = 0.9
horizon = 40

world_map = np.array([
    ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x'],
    ['x', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x'],
    ['x', ' ', ' ', ' ', 'x', ' ', ' ', ' ', 'x'],
    ['x', ' ', 'x', 'O', 'x', 'O', 'x', ' ', 'x'],
    ['x', ' ', ' ', 'O', 'x', 'O', ' ', ' ', 'x'],
    ['x', ' ', ' ', ' ', 'x', 'O', ' ', ' ', 'x'],
    ['x', ' ', ' ', 'G', 'x', 'O', ' ', ' ', 'x'],
    ['x', 'x', 'x', 'x', 'x', 'x', ' ', 'x', 'x'],
    ['x', 'x', 'x', 'x', 'x', 'x', ' ', 'x', 'x'],
    ['x', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x'],
    ['x', ' ', ' ', ' ', 'x', ' ', ' ', ' ', 'x'],
    ['x', ' ', 'x', 'O', 'x', 'O', 'x', ' ', 'x'],
    ['x', ' ', ' ', 'O', 'x', 'O', ' ', ' ', 'x'],
    ['x', ' ', ' ', ' ', 'x', 'O', ' ', ' ', 'x'],
    ['x', ' ', ' ', 'S', 'x', 'O', 'P', ' ', 'x'],
    ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x']])

mdp = GridWorld(
    world_map, gamma=gamma,
    rewards={'goal': 0, 'default': -1, 'puddle': -2, 'pit': -1},
    terminate_on_goal=False)

mdp.sample_trajectory(
    ValueIteration().fit(mdp), horizon=horizon, display=False)

state_feature_extractor = RectangularTileCoding(
    n_tilings=1, bounds=mdp.world_map.bounds.s, granularity=1, intercept=True)
feature_extractor = StateActionFeatureExtractor(
    state_feature_extractor, mdp.n_actions)

X = []
Y = []

for s in mdp.states:
    for a in mdp.actions:
        X.append(feature_extractor.as_vector(s, a))

        mdp.reset(s)
        s_prime, r = mdp.execute_action(a)
        Y.append(r)

from sklearn.linear_model import Ridge
model = Ridge(alpha=0.5)
model.fit(X, Y)
true_theta = model.coef_

linear_mdp = LinearRewardMDP(mdp, feature_extractor, true_theta)
theta_policy = ValueIteration().fit(linear_mdp)
linear_mdp.sample_trajectory(theta_policy, horizon=horizon, display=True)
