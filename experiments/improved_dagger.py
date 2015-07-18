import numpy as np
from mdp import UniformRandomPolicy, MDPPolicy
from pomdp import UniformRandomPolicy as POUniformRandomPolicy
from pomdp import BeliefStatePolicy

from learning_algorithm import LearningAlgorithm

from spectral_dagger.utils.math import geometric_sequence
from spectral_dagger.utils.math import laplace_smoothing
from spectral_dagger.mdp import LinearGibbsPolicy

from spectral_dagger.envs import ContinuousGridWorld
from spectral_dagger.envs import GridWorld
from spectral_dagger.mdp import ValueIteration
from spectral_dagger.function_approximation import RectangularTileCoding
from spectral_dagger.function_approximation import StateActionFeatureExtractor


class MixedPolicy(MDPPolicy):

    def __init__(self, p1, p2, beta, seed=None):
        """
        p1: the first policy
        p2: the second policy
        beta: the probability of choosing an action from the first policy
        """

        self.p1 = p1
        self.p2 = p2
        self.beta = beta
        self.rng = np.random.RandomState(seed)

    def reset(self, state):
        self.p1.reset(state)
        self.p2.reset(state)

    def update(self, action, state, reward=None):
        self.p1.update(action, state, reward)
        self.p2.update(action, state, reward)

    def get_action(self):
        p = self.rng.rand()
        if p < self.beta:
            a = self.p1.get_action()
        else:
            a = self.p2.get_action()

        return a

    def action_distribution(self, s):
        try:
            d1 = self.p1.action_distribution(s)
            d2 = self.p2.action_distribution(s)
            d = self.beta * d1 + (1 - self.beta) * d2
            assert sum(d) == 1
            return d

        except NotImplemented:
            raise NotImplemented(
                "No action distribution defined for this policy.")


class MixedExpertPolicy(MixedPolicy):

    def __init__(self, expert, p2, beta, seed):
        """
        The same functionality as MixedPolicy, except expert actions
        are recorded.

        p1: the first policy
        p2: the second policy
        beta: the probability of choosing an action from the first policy
        """
        super(MixedPolicy, self).__init__(expert, p2, beta, seed)
        self.expert_actions = []

    def reset(self, state):
        super(MixedPolicy, self).reset(state)
        self.expert_actions.append([])

    def get_action(self):
        expert_action = self.p1.get_action()
        self.expert_actions.append((self.p1.current_state, expert_action))

        if self.rng.rand() < self.beta:
            a = expert_action
        else:
            a = self.p2.get_action()

        return a


class ImprovedDaggerOptimizer(LearningAlgorithm):
    """
    Given a data-set consisting of a number of state-action pairs from the
    expert organized into trajectories.

    Data that we need:
    * n_iters pure expert trajectories
    * n_samples_per_iter examples of states and actions from the expert
    * lmbda, giving the weighting applied to the two terms in the optimization
    * L trajectories used (collected as part of the DAgger process) to use as_vector
    part the importance sampling
    * A differentiable classifier.

    Parameters
    ----------
    """

    def __init__(self, lmbda, classifier):

        self.lmbda = lmbda
        self.classifier = classifier

    def fit(self, expert_trajectories, expert_examples, training_trajectories):
        self.expert_trajectories = expert_trajectories
        self.expert_examples = expert_examples
        self.training_trajectories = training_trajectories

        # Do the optimization, return a policy

        # To begin an iteration: compute 
        # To compute the gradient:
        # 1. Compute action probabilities for all trajectories in the data set
        # at theta, the current point.
        # 2. 


def improved_dagger(
        mdp, expert, learning_alg, beta,
        n_iterations, n_samples_per_iter, horizon,
        initial_policy=None):
    """
    A version of DAgger which uses importance-sampled policy gradient in order
    to attempt to make use of distributional information available to it.
    """

    print "Running Improved DAgger..."

    if initial_policy is None:
        initial_policy = UniformRandomPolicy(mdp)

    policy = initial_policy

    policies = [initial_policy]

    trajectories = []
    expert_actions = []

    for i in range(n_iterations):
        b = beta.next()

        # Create a policy that is a mixture of our current policy and the expert policy
        mixed_policy = MixedExpertPolicy(expert, current_policy, b)

        print "Starting DAgger iteration ", i
        print "Beta for this iteration is ", b

        for j in range(n_samples_per_iter):

            tau = mdp.sample_trajectory(
                mixed_policy, horizon=horizon, display=False)

        policy = learning_alg.fit(mdp, states, expert_actions)

        policies.append(policy)

    print "Done DAgger."

    return policies

n_expert_trajectories = 5
smoothing = 1.0
delta = 0.1
gamma = 0.99
learning_rate = geometric_sequence(0.2, tau=33)

world_map = np.array([
    ['x', 'x', 'x', 'x', 'x', 'x', 'x', ' ', 'x'],
    ['x', 'G', ' ', 'S', ' ', ' ', ' ', ' ', 'x'],
    ['x', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'x'],
    ['x', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x']])

mdp = GridWorld(
    world_map, gamma=gamma,
    rewards={'goal': 0, 'default': -1, 'puddle': -5},
    terminate=False)

training_trajectories = []
previous_policies = []

# horizon
horizon = 10

# num dagger iterations
n_iters = 2

# num samples per iteration
n_samples_per_iter = 5

expert = ValueIteration().fit(mdp)

# sample expert trajectories
expert_action_data = defaultdict(list)

expert_trajectories = []
expert_examples = []
for n in range(n_expert_trajectories):
    trajectory = mdp.sample_trajectory(
        expert, horizon=horizon, display=False)
    expert_trajectories.append(trajectory)

    for s, a, r in trajectory:
        expert_action_data[s].append(a)
        expert_examples.append((s, a))

# estimate expert action distributions conditioned on state
expert_action_dist = {
    s: laplace_smoothing(smoothing, mdp.actions, expert_action_data[s])
    for s in mdp.states}

# Horizon is horizon. We go through horizon+1 states, but only take horizon actions. The first
# state is x_0, the first action is a_0. The last state is x_H, the last action
# is a_(horizon-1)

# For each trajectory and policy, need a running product

# trajectories x time
running_products = np.zeros(n_samples_per_iter * n_iters, horizon)

# For each state-action pair in each observed trajectory, store the log of the expert
neg_log_expert_distributions = np.zeros(n_samples_per_iter * n_iters, horizon)


def compute_weights(theta_prefix_probs, prefix_probs):
    # We have n_trajectories * horizon weights. These are obtained
    # by element-wise multiplication division between theta probabilities
    # and sigma probabilities
    weights = pi / sigma
    return weights


def Z(weights):
    # We have a Z for each time step. Time is the second axies of weights
    Z_theta = np.sum(weights, axis=0)
    return Z_theta


def J_tilde_theta(Z, weights, rewards):
    # weights and rewards both have dimensions (trajectories x horizon)
    # Z has dimension horizon. The result has dimension horizon
    J = np.sum(weights * rewards, axis=0) / Z
    return J


def reward(pi, log_expert):
    # pi and log_expert both n_trajectories * horizon
    return np.log(pi) - log_expert


def offset_reward(reward, J_tt, w_r):
    return reward - J_tt + w_r


def final_weights(weights, reward, Z):
    return weights * reward / 

# Going to use a linear Gibbs policy for now

# gradient:
gradients = np.zeros((n_trajectories, horizon, n_features))


for k, (s, a, r) in enumerate(trajectory):
    feature_vectors = np.array([
        policy.feature_extractor.as_vector(s, b)
        for b in mdp.actions])

    # need to do this for every trajectory and every time point
    feature_extractor.as_vector(s, a) - feature_vectors.T.dot(policy.action_distribution(s)))


# horizon
horizon = 3

# num dagger iterations
n_iters = 2

# num samples per iteration
n_samples_per_iter = 5


for i in range(n_iters):
    # Perform the optimization

    # Calculate the gradient
    for t in range(horizon):
        weights_t = []
        for tau in trajectories:
            weight = np.prod(current_policy[:t])
            weight /= 

            weights_t.append()
        Z_t = sum([])



    # Now samples some trajectories from the policy
    # that we have just found through the optimization
    for j in range(n_samples_per_iter):

