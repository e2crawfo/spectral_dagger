import numpy as np
import logging

from spectral_dagger.mdp import MDPPolicy

from sklearn.gaussian_process import GaussianProcess


class PILCO(MDPPolicy):

    def __init__(self, policy_class, *policy_args, **policy_kwargs):
        self.policy_class = policy_class
        self.policy_args = policy_args
        self.policy_kwargs = policy_kwargs

    def fit(
            self, mdp, horizon, feature_extractor,
            alpha, n_init_samples, n_samples_per_step, max_steps=np.inf,
            tol=1e-6, theta=None, mu=0.0):
        """
        Use the PILCO algorithm to find a high-reward agent
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
        n_init_samples: positive int
            Number of samples used to estimate initial model.
        n_samples_per_step: positive int
            Number of samples taken for each new parameter setting.
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

        self.log_file_name = 'PILCO.log'
        self.logger = logging.getLogger("PILCO")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.FileHandler(
            filename=self.log_file_name, mode='w'))

        policy = self.policy_class(
            mdp.actions, feature_extractor, theta,
            *self.policy_args, **self.policy_kwargs)

        # estimate initial model
        sample_trajectories = [
            mdp.sample_trajectory(
                policy, horizon, reset=True, display=False)
            for k in range(n_init_samples)]

        data = []
        targets = []
        for (s_0, a_0, r_0), (s_1, _, _) in zip(
                sample_trajectories, sample_trajectories[1:]):

            action_vector = np.zeros(mdp.n_actions)
            action_vector[a_0] = 1.0
            data.append(np.hstack(s_0, action_vector))
            targets.append(s_1 - s_0)

        targets = np.array(targets)

        gps = []
        for t in targets.T:
            # How to estimate the nugget?
            gp = GaussianProcess(
                corr='squared_exponential', theta0=1e-1, thetaL=1e-3, thetaU=1,
                nugget=1, random_start=100)
            gp.fit(data, t)
            gps.append(gp)

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

