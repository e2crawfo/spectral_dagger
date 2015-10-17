import numpy as np
import logging

from spectral_dagger.mdp import MDPPolicy

from sklearn import linear_model


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
            sample_trajectories, returns = self.get_sample_trajectories(
                n_samples, mdp, policy, horizon)

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

            print "Average return: %f" % np.mean(returns)
            print (
                "Updated theta with alpha: %s, theta norm: %s" % (alpha, norm))

            self.logger.info("*" * 20 + "Iteration: %s" % n_steps)
            self.logger.info("Gradient step")
            self.logger.info("Step norm: %s" % norm)
            self.logger.info("Alpha: %s" % alpha)
            self.logger.info("Average return: %f", np.mean(returns))

            n_steps += 1

        return policy

    @staticmethod
    def get_sample_trajectories(n_samples, mdp, policy, horizon):
        sample_trajectories = [
            mdp.sample_trajectory(
                policy, horizon, reset=True, display=False)
            for k in range(n_samples)]

        returns = [
            sum(r for (_, _, r) in tau) for tau in sample_trajectories]

        return sample_trajectories, returns

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


class TrajectoryData(object):
    """ Object storing relevant features of a trajectory """
    def __init__(self, features, state_0, total_reward):
        self.features = features
        self.state_0 = state_0
        self.total_reward = total_reward


class eNAC(REINFORCE):

    def __init__(self, policy_class, *policy_args, **policy_kwargs):
        self.policy_class = policy_class
        self.policy_args = policy_args
        self.policy_kwargs = policy_kwargs

    @staticmethod
    def get_sample_trajectories(n_samples, mdp, policy, horizon):
        """
        Return data summarizing sampled trajectories.

        The gradient estimate does not require the complete trajectories,
        so don't bother storing them.
        """

        print "Sampling %d trajectories." % n_samples
        trajectory_data = []
        returns = []

        for i in range(n_samples):
            if i % 100 == 0:
                print "Sampling trajectory %d" % i

            tau = mdp.sample_trajectory(
                policy, horizon, reset=True, display=False)

            tau_data = TrajectoryData(
                features=np.zeros(policy.theta.size),
                state_0=np.copy(tau[0][0]),
                total_reward=0)

            for t, (s, a, r) in enumerate(tau):
                tau_data.features += (
                    (mdp.gamma ** t) * policy.gradient_log(s, a))
                tau_data.total_reward += r

            trajectory_data.append(tau_data)
            returns.append(tau_data.total_reward)

        return trajectory_data, returns

    @staticmethod
    def estimate_gradient(sample_trajectories, policy):
        """
        Estimate gradient direction from sample trajectories.

        Returns: unit vector in gradient direction, gradient norm
        """
        X_value = []
        Y_value = []

        for tau_data in sample_trajectories:
            X_value.append(tau_data.state_0)
            Y_value.append(tau_data.total_reward)

        X_value = np.array(X_value)
        Y_value = np.array(Y_value)

        value_function = linear_model.LinearRegression(
            fit_intercept=True,
            normalize=False,
            copy_X=False)

        value_function.fit(X_value, Y_value)

        X = []
        Y = []

        # Now perform the regression
        for tau_data in sample_trajectories:
            X.append(tau_data.features)
            Y.append(
                tau_data.total_reward
                + value_function.predict(tau_data.state_0))

        X = np.array(X)
        Y = np.array(Y)

        model = linear_model.LinearRegression(
            fit_intercept=False,
            normalize=False,
            copy_X=False)

        model.fit(X, Y)
        gradient = np.copy(model.coef_)

        norm = np.linalg.norm(gradient)
        if norm > 0:
            gradient /= norm

        return gradient, norm


# if __name__ == "__main__":
#     from spectral_dagger.envs import ContinuousGridWorld
#     from spectral_dagger.mdp import LinearGibbsPolicy
#     from spectral_dagger.function_approximation import RectangularTileCoding
#     from spectral_dagger.function_approximation import StateActionFeatureExtractor
#     from spectral_dagger.utils.math import p_sequence
# 
#     world_map = np.array([
#         ['x', 'x', 'x', 'x', 'x', 'x', 'x'],
#         ['x', 'P', 'P', 'P', 'P', 'G', 'x'],
#         ['x', 'P', ' ', ' ', ' ', ' ', 'x'],
#         ['x', 'P', ' ', 'P', ' ', 'P', 'x'],
#         ['x', 'P', ' ', 'P', ' ', 'P', 'x'],
#         ['x', 'P', ' ', 'P', ' ', 'P', 'x'],
#         ['x', 'P', ' ', 'P', ' ', 'P', 'x'],
#         ['x', 'x', 'x', 'x', 'x', 'x', 'x']])
#     world_map = np.array([
#         ['x', 'x', 'x', 'x'],
#         ['x', ' ', 'G', 'x'],
#         ['x', ' ', ' ', 'x'],
#         ['x', ' ', ' ', 'x'],
#         ['x', ' ', ' ', 'x'],
#         ['x', ' ', ' ', 'x'],
#         ['x', ' ', ' ', 'x'],
#         ['x', 'x', 'x', 'x']])
# 
# 
#     horizon = 400
#     alpha = p_sequence(start=3.0, p=0.6)
#     gamma = 0.9
# 
#     mdp = ContinuousGridWorld(
#         world_map, gamma=gamma, speed=0.4,
#         rewards={'goal': 0, 'default': -1, 'puddle': -5},
#         terminate_on_goal=True)
# 
#     state_feature_extractor = RectangularTileCoding(
#         n_tilings=2, extent=mdp.world_map.bounds.s,
#         tile_dims=0.3, intercept=True)
# 
#     feature_extractor = StateActionFeatureExtractor(
#         state_feature_extractor, mdp.n_actions)
# 
#     learner = REINFORCE(LinearGibbsPolicy)
#     policy = learner.fit(
#         mdp, horizon, feature_extractor, alpha,
#         n_samples=2, max_steps=200)
# 
#     for i in range(10):
#         mdp.sample_trajectory(policy, horizon, reset=True, display=0.1)

if __name__ == "__main__":
    from spectral_dagger.envs import ContinuousGridWorld
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
    world_map = np.array([
        ['x', 'x', 'x', 'x'],
        ['x', ' ', 'G', 'x'],
        ['x', ' ', ' ', 'x'],
        ['x', ' ', ' ', 'x'],
        ['x', ' ', ' ', 'x'],
        ['x', ' ', ' ', 'x'],
        ['x', ' ', ' ', 'x'],
        ['x', 'x', 'x', 'x']])

    horizon = 100
    alpha = p_sequence(start=3.0, p=0.6)
    gamma = 0.9

    mdp = ContinuousGridWorld(
        world_map, gamma=gamma, speed=0.4,
        rewards={'goal': 0, 'default': -1, 'puddle': -5},
        terminate_on_goal=True)

    state_feature_extractor = RectangularTileCoding(
        n_tilings=1, extent=mdp.world_map.bounds.s,
        tile_dims=0.3, intercept=True)

    feature_extractor = StateActionFeatureExtractor(
        state_feature_extractor, mdp.n_actions)

    learner = eNAC(LinearGibbsPolicy)
    policy = learner.fit(
        mdp, horizon, feature_extractor, alpha,
        n_samples=int(feature_extractor.n_features/8.0), max_steps=200)

    for i in range(10):
        mdp.sample_trajectory(policy, horizon, reset=True, display=0.1)
