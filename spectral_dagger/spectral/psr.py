import numpy as np
from scipy.sparse import csr_matrix
from sklearn.utils.extmath import randomized_svd
import logging

from spectral_dagger import sample_episodes, LearningAlgorithm, Space, Policy
from spectral_dagger.spectral import hankel


logger = logging.getLogger(__name__)


class PredictiveStateRep(object):
    def __init__(self, b_0, b_inf, B_o):
        self.b_0 = b_0
        self.b_inf = b_inf

        self.B_o = B_o

        self.observations = B_o.keys()
        self.n_observations = len(self.observations)

    def action_space(self):
        return None

    def observation_space(self):
        return Space(set(self.observations), "ObsSpace")

    def reset(self):
        self.b = self.b_0.copy()

    def update(self, action=None, obs=None):
        """ Update state upon seeing an observation. """
        B_o = self.B_o[obs]
        numer = self.b.dot(B_o)
        denom = numer.dot(self.b_inf)

        self.b = numer / denom

    def get_obs_prob(self, o):
        """ Get probability of observation for next time step.  """
        prob = self.b.dot(self.B_o[o]).dot(self.b_inf)
        return np.clip(prob, np.finfo(float).eps, 1)

    def get_prediction(self):
        """ Get observation with highest prob for next time step . """
        predict = lambda o: self.get_obs_prob(o)
        return max(self.observations, key=predict)

    def get_obs_rank(self, o):
        """ Get probability rank of observation for next time step. """
        probs = np.array(
            [self.get_obs_prob(obs) for obs in self.observations])

        return (
            np.count_nonzero(probs > self.get_obs_prob(o)),
            np.count_nonzero(probs < self.get_obs_prob(o)))

    def get_seq_prob(self, seq):
        """ Get probability of sequence given current state. """

        b = self.b.copy()
        for o in seq:
            b = b.dot(self.B_o[o])

        prob = b.dot(self.b_inf)

        return np.clip(prob, np.finfo(float).eps, 1)

    def get_delayed_seq_prob(self, seq, t):
        """ Get probability of observing sequence at a delay of ``t``.

        get_delayed_seq_prob(seq, 0) is equivalent to get_seq_prob(seq).

        """
        b = self.b.copy()

        for i in range(t):
            b = b.dot(self.B)

        for o in seq:
            b = b.dot(self.B_o[o])

        prob = b.dot(self.b_inf)

        return np.clip(prob, np.finfo(float).eps, 1)

    def get_seq_state(self, seq, b=None):
        """ Get state obtained if sequence observed from state `b`. """
        old_b = self.b

        self.b = b.copy() if b else self.b.copy()
        for o in seq:
            self.filter(o)

        new_b = self.b
        self.b = old_b

        return new_b

    def get_WER(self, test_data):
        """ Get word error rate for the test data. """
        errors = 0
        n_predictions = 0

        for seq in test_data:
            self.reset()

            for o in seq:
                prediction = self.get_prediction()

                if prediction != o:
                    errors += 1

                self.update(o)

                n_predictions += 1

        return errors/float(n_predictions)

    def get_log_likelihood(self, test_data, base=2):
        """ Get average log likelihood for the test data.

        Done using `get_obs_prob` instead of `get_seq_prob` to avoid
        problems related to vanishing probabilities.

        """
        llh = 0.0

        for seq in test_data:
            seq_llh = 0.0

            self.reset()

            for o in seq:
                if base == 2:
                    seq_llh += np.log2(self.get_obs_prob(o))
                else:
                    seq_llh += np.log(self.get_obs_prob(o))

                self.update(o)

            llh += seq_llh

        return llh / len(test_data)

    def get_perplexity(self, test_data, base=2):
        """ Get model perplexity on the test data.  """

        return 2**(-self.get_log_likelihood(test_data, base=base))


class SpectralPSR(PredictiveStateRep):
    def __init__(self, observations):
        self.b_0 = None
        self.b_inf = None
        self.B_o = None

        self.observations = observations
        self.n_observations = len(self.observations)

    def fit(
            self, data, n_components, estimator='prefix',
            basis=None, svd=None, hankels=None):
        """ Fit a PSR to the given data using a spectral algorithm.

        data: list of list
            Each sublist is a list of observations constituting a trajectory.
        n_components: int
            Number of dimensions in feature space.
        estimator: string
            'string', 'prefix', or 'substring'.
        basis: length-2 tuple
            Contains prefix and suffix dictionaries.
        svd: length-3 tuple
            Contains U, S, V^T, the SVD of a Hankel matrix. If provided, then
            computing the SVD of the estimated Hankel matrix is skipped.
        hankels: length-4 tuple
            Contains hp, hs, hankel, symbol_hankels. If provided, then
            estimating the Hankel matrices is skipped. If provided, then
            a basis must also be provided.

        """
        if hankels:
            if not basis:
                raise ValueError(
                    "If `hankels` provided, must also provide a basis.")

            hp, hs, hankel_matrix, symbol_hankels = hankels
        else:
            if not basis:
                logger.debug("Generating basis...")
                basis = hankel.top_k_basis(data, max_basis_size, estimator)

            logger.debug("Estimating Hankels...")
            hankels = hankel.estimate_hankels(
                data, basis, self.observations, estimator)

            # Note: all hankels are scipy csr matrices
            hp, hs, hankel_matrix, symbol_hankels = hankels

        self.basis = basis

        self.hankel = hankel_matrix
        self.symbol_hankels = symbol_hankels

        n_components = min(n_components, hankel_matrix.shape[0])

        if svd:
            U, S, VT = svd
        else:
            logger.debug("Performing SVD...")
            n_oversamples = 10
            n_iter = 5

            # H = U S V^T
            U, S, VT = randomized_svd(
                hankel_matrix, 50, n_oversamples, n_iter)

        V = VT.T

        U = U[:, :n_components]
        V = V[:, :n_components]
        S = np.diag(S[:n_components])

        # P^+ = (HV)^+ = (US)^+ = S^+ U+ = S^-1 U.T
        P_plus = csr_matrix(np.linalg.pinv(S).dot(U.T))

        # S^+ = (V.T)^+ = V
        S_plus = csr_matrix(V)

        logger.debug("Computing operators...")
        self.B_o = {}
        for o in self.observations:
            B_o = P_plus.dot(symbol_hankels[o]).dot(S_plus)
            self.B_o[o] = B_o.toarray()

        # b_0 S = hs => b_0 = hs S^+
        self.b_0 = hs.dot(S_plus)
        self.b_0 = self.b_0.toarray()[0, :]

        # P b_inf = hp => b_inf = P^+ hp
        self.b_inf = P_plus.dot(hp)
        self.b_inf = self.b_inf.toarray()[:, 0]

        self.B = sum(self.B_o.values())

        # See Lemma 6.1.1 in Borja Balle's thesis
        I_minus_B = np.eye(n_components) - self.B
        if estimator != 'substring':
            I_minus_B_inv = np.linalg.pinv(I_minus_B)

        if estimator == 'string':
            self.b_inf_string = self.b_inf
            self.b_inf = I_minus_B_inv.dot(self.b_inf_string)

            self.b_0_substring = self.b_0.dot(I_minus_B_inv)

        elif estimator == 'prefix':
            self.b_inf_string = I_minus_B.dot(self.b_inf)

            self.b_0_substring = self.b_0.dot(I_minus_B_inv)

        elif estimator == 'substring':
            self.b_inf_string = I_minus_B.dot(self.b_inf)

            self.b_0_substring = self.b_0
            self.b_0 = self.b_0_substring.dot(I_minus_B)

        else:
            raise ValueError("Unknown Hankel estimator name: %s." % estimator)

        self.reset()


class CompressedPSR(PredictiveStateRep):
    def __init__(self, observations):
        self.b_0 = None
        self.b_inf = None
        self.B_o = None

        self.observations = observations
        self.n_observations = len(self.observations)

    def fit(
            self, data, n_components,
            basis=None, phi=None, hankels=None):
        """ Fit a PSR to the given data using a compression algorithm. """

        self.basis = basis
        self.n_components = n_components

        prefix_dict, suffix_dict = basis

        if phi is None:
            phi = self.model_rng.randn(len(suffix_dict), n_components)
            phi *= 1. / np.sqrt(n_components)

        self.phi = phi

        hp = np.zeros(len(prefix_dict))

        proj_hankel = np.zeros((len(prefix_dict), n_components))
        proj_sym_hankels = {}
        for obs in self.observations:
            proj_sym_hankels[obs] = np.zeros((len(prefix_dict), n_components))

        self.b_0 = np.zeros(n_components)

        for seq in data:
            for i in range(len(seq)+1):
                prefix = tuple(seq[:i])

                if prefix in suffix_dict:
                    self.b_0 += phi[suffix_dict[prefix], :]

                if prefix not in prefix_dict:
                    continue

                prefix_idx = prefix_dict[prefix]
                hp[prefix_idx] += 1.0

                for j in range(i, len(seq)+1):
                    suffix = tuple(seq[i:j])

                    if suffix in suffix_dict:
                        suffix_idx = suffix_dict[suffix]
                        proj_hankel[prefix_idx, :] += phi[suffix_idx, :]

                    if suffix and suffix[1:] in suffix_dict:
                        o = suffix[0]
                        suffix_idx = suffix_dict[suffix[1:]]

                        proj_sym_hankel = proj_sym_hankels[o]
                        proj_sym_hankel[prefix_idx, :] += phi[suffix_idx, :]

        n_samples = float(len(data))

        self.b_0 /= n_samples
        proj_hankel /= n_samples
        hp /= n_samples

        inv_proj_hankel = np.linalg.pinv(proj_hankel)

        self.B_o = {}
        for o in self.observations:
            self.B_o[o] = inv_proj_hankel.dot(proj_sym_hankels[o] / n_samples)

        self.b_inf = inv_proj_hankel.dot(hp)

        self.reset()


class SpectralPSRWithActions(object):
    def __init__(self, actions, observations, max_dim=80):

        self.n_actions = len(actions)
        self.actions = actions

        self.n_observations = len(observations)
        self.observations = observations

        self.B_ao = {}
        self.max_dim = max_dim

    def action_space(self):
        return Space(set(self.actions), "ActionSpace")

    def observation_space(self):
        return Space(set(self.observations), "ObsSpace")

    def fit(self, data, max_basis_size, n_components, use_naive=False):
        """
        data should be a list of lists. Each sublist corresponds to a
        trajectory.  Each entry of the trajectory should be a 2-tuple,
        giving the action followed by the observation.
        """

        logger.debug("Generating basis...")
        basis = hankel.top_k_basis(data, max_basis_size)

        logger.debug("Estimating hankels...")

        # Note: all matrices returned by construct_hankels are csr_matrices
        if use_naive:
            logger.debug("...using naive estimator...")
            hankels = hankel.construct_hankels_with_actions(
                data, basis, self.actions, self.observations)

            hp, hs, hankel_matrix, symbol_hankels = hankels
        else:
            logger.debug("...using robust estimator...")
            hankels = hankel.construct_hankels_with_actions_robust(
                data, basis, self.actions, self.observations)

            hp, hs, hankel_matrix, symbol_hankels = hankels

        self.hankel = hankel_matrix
        self.symbol_hankels = symbol_hankels

        n_components = min(n_components, hankel_matrix.shape[0])

        logger.debug("Performing SVD...")
        n_oversamples = 10
        n_iter = 5

        # H = U S V^T
        U, S, VT = randomized_svd(
            hankel_matrix, self.max_dim, n_oversamples, n_iter)

        V = VT.T

        U = U[:, :n_components]
        V = V[:, :n_components]
        S = np.diag(S[:n_components])

        # P^+ = (HV)^+ = (US)^+ = S^+ U+ = S^-1 U.T
        P_plus = csr_matrix((np.linalg.pinv(S)).dot(U.T))

        # S^+ = (V.T)^+ = V
        S_plus = csr_matrix(V)

        logger.debug("Computing operators...")

        for pair in symbol_hankels:
            symbol_hankel = P_plus.dot(symbol_hankels[pair])
            symbol_hankel = symbol_hankel.dot(S_plus)
            self.B_ao[pair] = symbol_hankel.toarray()

        # computing stopping and starting vectors

        # P b_inf = hp => b_inf = P^+ hp
        self.b_inf = P_plus.dot(hp)
        self.b_inf = self.b_inf.toarray()[:, 0]

        # See Lemma 6.1.1 in Borja's thesis
        B = sum(self.B_ao.values())
        self.b_inf = np.linalg.pinv(np.eye(n_components)-B).dot(self.b_inf)

        # b_0 S = hs => b_0 = hs S^+
        self.b_0 = hs.dot(S_plus)
        self.b_0 = self.b_0.toarray()[0, :]

        self.reset()

        return self.b_0, self.B_ao, self.b_inf

    def update(self, action, obs):
        """Update state upon seeing an action observation pair"""
        B_ao = self.B_ao[action, obs]
        numer = self.b.dot(B_ao)
        denom = numer.dot(self.b_inf)

        self.b = numer / denom

    def reset(self):
        self.b = self.b_0.copy()

    def get_prediction(self, action):
        """
        Return the symbol that the model expects next,
        given that action is executed.
        """
        predict = lambda a: (lambda o: self.get_obs_prob(a, o))
        return max(self.observations, key=predict(action))

    def get_obs_prob(self, a, o):
        """
        Returns the probablilty of observing o given
        we take action a in the current state.
        """
        prob = self.b.dot(self.B_ao[(a, o)]).dot(self.b_inf)

        return np.clip(prob, np.finfo(float).eps, 1)

    def get_obs_rank(self, a, o):
        """
        Get the rank of the given observation, in terms of probability,
        given we take action a in the current state.
        """
        probs = np.array(
            [self.get_obs_prob(a, obs) for obs in self.observations])

        return (
            np.count_nonzero(probs > self.get_obs_prob(a, o)),
            np.count_nonzero(probs < self.get_obs_prob(a, o)))

    def get_seq_prob(self, seq):
        """Returns the probability of a sequence given current state"""

        state = self.b
        for (a, o) in seq:
            state = state.dot(self.B_ao[(a, o)])

        prob = state.dot(self.b_inf)

        return np.clip(prob, np.finfo(float).eps, 1)

    def get_state_for_seq(self, seq, initial_state=None):
        """Returns the probability of a sequence given current state"""

        state = self.b.T if initial_state is None else initial_state

        for (a, o) in seq:
            state = state.dot(self.B_ao[(a, o)])

        return state / state.dot(self.b_inf)

    def get_WER(self, test_data):
        """Returns word error rate for the test data"""
        errors = 0
        n_predictions = 0

        for seq in test_data:
            self.reset()

            for a, o in seq:
                prediction = self.get_prediction(a)

                if prediction != o:
                    errors += 1

                self.update(a, o)

                n_predictions += 1

        return errors/float(n_predictions)

    def get_log_likelihood(self, test_data, base=2):
        """Returns average log likelihood for the test data"""
        llh = 0

        for seq in test_data:
            seq_llh = 0

            self.reset()

            for (a, o) in seq:
                if base == 2:
                    seq_llh += np.log2(self.get_obs_prob(a, o))
                else:
                    seq_llh += np.log(self.get_obs_prob(a, o))

                self.update(a, o)

            llh += seq_llh

        return llh / len(test_data)


class SpectralClassifier(LearningAlgorithm):
    """
    A learning algorithm which learns to select actions in a
    POMDP setting based on observed trajectories and expert actions.
    Uses the observed trajectories to learn a PSR for the POMDP that
    gave rise to the trajectories, and then uses a classifier to learn
    a mapping between states of the PSR to expert actions.

    Parameters
    ----------
    predictor: any
        Must implement the sklearn Estimator and Predictor interfaces.
        http://scikit-learn.org/stable/developers/contributing.html

    max_basis_size: int
        The maximum size of the basis for the spectral learner.

    n_components: int
        The number of components to use in the spectral learner.

    """
    def __init__(self, predictor, max_basis_size=500, n_components=50):

        self.predictor = predictor
        self.max_basis_size = max_basis_size
        self.n_components = n_components

    def fit(self, pomdp, trajectories, actions):
        """ Returns a policy trained on the given data.

        Parameters
        ----------
        pomdp: POMDP
            The pomdp that the data was generated from.

        trajectories: list of lists of action-observation pairs
            Each sublist corresponds to a trajectory.

        actions: list of lists of actions
            Each sublist contains the actions generated by the expert in
            response to trajectories. The the j-th entry of the i-th sublist
            gives the action chosen by the expert in response to the first
            j-1 action-observation pairs in the i-th trajectory in
            `trajectories`. The actions in this data structure are not
            necessarily the same as the actions in `trajectories`.

        """
        self.psr = SpectralPSRWithActions(pomdp.actions, pomdp.observations)

        self.psr.fit(trajectories, self.max_basis_size, self.n_components)

        psr_states = []
        flat_actions = []

        for t, response_actions in zip(trajectories, actions):
            self.psr.reset()

            for (a, o), response_action in zip(t, response_actions):
                psr_states.append(self.psr.b)
                flat_actions.append(response_action)

                self.psr.update(a, o)

        # Most sklearn predictors operate on strings or numbers
        action_lookup = {str(a): a for a in set(flat_actions)}
        str_actions = [str(a) for a in flat_actions]

        self.predictor.fit(psr_states, str_actions)

        def f(psr_state):
            psr_state = psr_state.reshape(1, -1)
            action_string = self.predictor.predict(psr_state)[0]
            return action_lookup[action_string]

        return SpectralPolicy(self.psr, f)


class SpectralPolicy(Policy):
    def __init__(self, psr, f):
        self.psr = psr
        self.f = f

    @property
    def action_space(self):
        return self.psr.action_space

    @property
    def observation_space(self):
        return self.psr.observation_space

    def reset(self, init_dist=None):
        if init_dist is not None:
            raise Exception(
                "Cannot supply initialization distribution to PSR. "
                "Only works with the initialization distribution "
                "on which it was trained.")

        self.psr.reset()

    def update(self, action, observation, reward=None):
        self.psr.update(action, observation)

    def get_action(self):
        return self.f(self.psr.b)


if __name__ == "__main__":
    import grid_world

    # Sample a bunch of trajectories, run the learning algorithm on them
    n_trajectories = 20000
    horizon = 3
    n_components = 40
    max_basis_size = 1000
    max_dim = 150

    world = np.array([
        ['x', 'x', 'x', 'x', 'x'],
        ['x', ' ', ' ', 'G', 'x'],
        ['x', ' ', ' ', ' ', 'x'],
        ['x', ' ', 'x', ' ', 'x'],
        ['x', ' ', ' ', ' ', 'x'],
        ['x', 'S', ' ', ' ', 'x'],
        ['x', 'x', 'x', 'x', 'x']]
    )

    n_colors = 2

    pomdp = grid_world.EgoGridWorld(n_colors, world)

    exploration_policy = POMDPPolicy()
    exploration_policy.fit(pomdp)

    trajectories = []

    print "Sampling trajectories..."
    trajectories = sample_episodes(
        n_trajectories, pomdp, exploration_policy, horizon)

    for use_naive in [True, False]:
        print "Training model..."

        psr = SpectralPSRWithActions(
            pomdp.actions, pomdp.observations, max_dim)

        b_0, B_ao, b_inf = psr.fit(
            trajectories, max_basis_size, n_components, use_naive)

        test_length = 10
        n_tests = 2000

        n_below = []
        top_three_count = 0

        print "Running tests..."

        display = False

        for t in range(n_tests):

            if display:
                print "\nStart test"
                print "*" * 20

            exploration_policy.reset()
            psr.reset()
            pomdp.reset()

            for i in range(test_length):
                action = exploration_policy.get_action()
                predicted_obs = psr.get_prediction(action)

                pomdp_string = str(pomdp)

                actual_obs, _ = pomdp.update(action)

                rank = psr.get_obs_rank(action, actual_obs)

                n_below.append(rank[1])
                if rank[0] < 3:
                    top_three_count += 1

                psr.update(action, actual_obs)

                exploration_policy.update(action, actual_obs)

                if display:
                    print "\nStep %d" % i
                    print "*" * 20
                    print pomdp_string
                    print "Chosen action: ", action
                    print "Predicted observation: ", predicted_obs
                    print "Actual observation: ", actual_obs
                    print "PSR Rank of Actual Observation: ", rank

        print (
            "Average num below: ", np.mean(n_below),
            "of", len(pomdp.observations))
        print "Probability in top 3: %f" % (
            float(top_three_count) / (test_length * n_tests))

        n_test_trajectories = 40
        test_trajectories = []

        print "Sampling test trajectories for WER..."
        trajectories = sample_episodes(
            n_test_trajectories, pomdp, exploration_policy, horizon)

        print "Word error rate: ", psr.get_WER(test_trajectories)

        llh = psr.get_log_likelihood(test_trajectories, base=2)
        print "Average log likelihood: ", llh
        print "Perplexity: ", 2**(-llh)
