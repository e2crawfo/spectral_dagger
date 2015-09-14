import numpy as np

from mdp import MDPPolicy
from td import LinearGradientTD

# Dummy form of the policy gradient: sample a bunch of trajectories
# using current policy, evaluate (gradient of policy/policy) * value
# of objective function

class SimplePolicyGradient(MDPPolicy):

    def __init__(self, mdp):
        pass

    def sample

class PolicyGradient(MDPPolicy):

    """
    Estimate value function, estimate state-value function,
    using linear Gibbs (softmax) distribution for choosing actions
    for now.
    """

    def __init__(self, mdp, feature_extractor, alpha, L=0):
        self.feature_extractor = feature_extractor
        self.theta = np.zeros(feature_extractor.n_features)

        self.Q_estimators = [
            LinearGradientTD(mdp, feature_extractor, alpha, L)
            for a in mdp.actions]

        self.V_estimator = LinearGradientTD(mdp, feature_extractor, alpha, L)

    def reset(self):
        for q in self.Q_estimators:
            q.reset()

        self.V_estimator.reset()

    def update(self, action, state, reward=None):


    def get_action(self):
        probabilities = np.exp()
        sample = np.random.multinomial(1, self._T[action, self.current_state])
        self.current_state = self.states[np.where(sample > 0)[0][0]]

        self.theta += self.alpha * delta * self.eligibility_trace


    def V(self, state):
        pass

    def Q(self, state, action):
        pass