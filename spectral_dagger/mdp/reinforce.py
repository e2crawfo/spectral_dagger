import numpy as np
import logging

from spectral_dagger.mdp import MDPPolicy


class REINFORCE(MDPPolicy):

    def __init__(self, policy_class, *policy_args, **policy_kwargs):
        self.policy_class = policy_class
        self.policy_args = policy_args
        self.policy_kwargs = policy_kwargs

    def fit(
            self, mdp, horizon, feature_extractor,
            alpha, n_samples, max_steps=np.inf, tol=1e-6, theta=None,
            mu=0.0):
        """
        Use the REINFORCE algorithm to find a high-reward agent
        in the given MDP.

        Parameters
        ----------
        mdp: MDP instance
            The MDP we are learning on.
        horizon: positive int
            Number maximum number of steps per episode.
        feature_extractor: FeatureExtractor instance
            Obtain features from states.
        alpha: iterable
            Sequence of learning rates.
        n_samples: positive int
            Number of samples per gradient estimate.
        max_steps: positive int
            Maximum number of gradient ascent steps to execute.
        tol: non-negative float, optional
            Will stop optimizing once the learning rates
            are below this value.
        theta: 1-D numpy array, optional
            Initial guess for parameters.
        mu: non-negative float, optional
            Momentum parameter.

        """
        self.mdp = mdp
        self.horizon = horizon
        self.feature_extractor = feature_extractor

        if theta is None:
            theta = np.zeros(feature_extractor.n_features)
        else:
            theta = theta.flatten()
            assert (
                theta.ndim == 1 and
                theta.shape[0] == feature_extractor.n_features)

        self.log_file_name = 'REINFORCE.log'
        self.logger = logging.getLogger("REINFORCE")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.FileHandler(
            filename=self.log_file_name, mode='w'))

        policy = self.policy_class(
            mdp.actions, feature_extractor, theta,
            *self.policy_args, **self.policy_kwargs)

        _alpha = alpha
        alpha = _alpha.next()

        gradient = np.zeros(feature_extractor.n_features)

        n_steps = 0
        while alpha > tol and n_steps < max_steps:
            sample_trajectories = [
                mdp.sample_trajectory(
                    policy, horizon, reset=True, display=False)
                for k in range(n_samples)]

            current_gradient, norm = (
                self.estimate_gradient(sample_trajectories, policy))

            # implement momentum
            gradient = mu * gradient + current_gradient
            policy.theta += alpha * gradient

            norm = np.linalg.norm(gradient)
            if norm > 0:
                gradient /= norm

            policy.theta += alpha * gradient

            if hasattr(_alpha, 'next'):
                try:
                    next_alpha = _alpha.next()
                    alpha = next_alpha
                except StopIteration:
                    pass

            returns = [
                sum(r for (_, _, r) in tau) for tau in sample_trajectories]
            for i, r in enumerate(returns):
                print "Return from iter %d:  %f" % (i, r)

            print (
                "Updated theta with alpha: %s, theta norm: %s" % (alpha, norm))

            self.logger.info("*" * 20 + "Iteration: %s" % n_steps)
            self.logger.info("Gradient step")
            self.logger.info("Step norm: %s" % norm)
            self.logger.info("Alpha: %s" % alpha)
            for i, r in enumerate(returns):
                self.logger.info("Return from iter %d:  %f", i, r)

            n_steps += 1

        return policy

    @staticmethod
    def estimate_gradient(sample_trajectories, policy):
        """
        Estimate gradient direction from sample trajectories.

        Returns: unit vector in gradient direction, gradient norm
        """

        cumulative_reward = []
        horizon = max(len(tau) for tau in sample_trajectories)

        for tau in sample_trajectories:
            r = [r for (_, _, r) in tau]
            r.extend([0.0] * (horizon - len(r)))
            cumulative_reward.append(sum(r) - np.cumsum(r))

        cumulative_reward = np.array(cumulative_reward)
        average_cumu_reward = np.mean(cumulative_reward, axis=0)
        baselined_reward = cumulative_reward - average_cumu_reward

        # Calculate gradient estimate
        gradient = np.zeros(policy.feature_extractor.n_features)

        for br, tau in zip(baselined_reward, sample_trajectories):
            for r, (s, a, _) in zip(br, tau):
                gradient += policy.gradient_log(s, a) * r

        norm = np.linalg.norm(gradient)
        if norm > 0:
            gradient /= norm

        return gradient, norm


if __name__ == "__main__":
    from spectral_dagger.envs import ContinuousGridWorld
    from spectral_dagger.envs import GridWorld
    from spectral_dagger.mdp import LinearGibbsPolicy
    from spectral_dagger.function_approximation import RectangularTileCoding
    from spectral_dagger.function_approximation import StateActionFeatureExtractor
    from spectral_dagger.utils.math import p_sequence

    world_map = np.array([
        ['x', 'x', 'x', 'x', 'x', 'x', 'x'],
        ['x', 'P', 'P', 'P', 'P', 'G', 'x'],
        ['x', 'P', ' ', ' ', ' ', ' ', 'x'],
        ['x', 'P', ' ', 'P', ' ', 'P', 'x'],
        ['x', 'P', ' ', 'P', ' ', 'P', 'x'],
        ['x', 'P', ' ', 'P', ' ', 'P', 'x'],
        ['x', 'P', ' ', 'P', ' ', 'P', 'x'],
        ['x', 'x', 'x', 'x', 'x', 'x', 'x']])

    horizon = 400
    alpha = p_sequence(start=3.0, p=0.6)
    gamma = 0.9

    mdp = ContinuousGridWorld(
        world_map, gamma=gamma, speed=0.4,
        rewards={'goal': 0, 'default': -1, 'puddle': -5},
        terminate_on_goal=True)

    state_feature_extractor = RectangularTileCoding(
        n_tilings=2, extent=mdp.world_map.bounds.s,
        tile_dims=0.3, intercept=True)

    feature_extractor = StateActionFeatureExtractor(
        state_feature_extractor, mdp.n_actions)

    learner = REINFORCE(LinearGibbsPolicy)
    policy = learner.fit(
        mdp, horizon, feature_extractor, alpha,
        n_samples=2, max_steps=200)

    for i in range(10):
        mdp.sample_trajectory(policy, horizon, reset=True, display=0.1)
