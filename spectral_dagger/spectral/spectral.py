import numpy as np
from scipy.sparse import csr_matrix

from sklearn.utils.extmath import randomized_svd

from spectral_dagger.spectral import hankel
from spectral_dagger.learning_algorithm import LearningAlgorithm
from spectral_dagger.pomdp import POMDPPolicy


class SpectralPSR(object):
    def __init__(self, observations, max_dim=80):
        self.n_observations = len(observations)
        self.observations = observations

        self.B_o = {}
        self.max_dim = max_dim

    def fit(self, data, n_components, max_basis_size=np.inf, basis=None):
        """
        Some strange differences between Hsu et al. and Borka's work.
        HSU uses strings of length 2 for H, but strings of length 3 for
        H_sigma. So we need to have strings of different length according to that,
        and the same should work Borja. But...then the probabilities that are stored
        in the matrices aren't really string probabilities. They are prefix probabilities.

        But Borja MODIFIES the original spectral algorithm to use a different normalization
        vector in order to deal properly with prefixes.

        data should be a list of lists. Each sublist is a list of
        observations constituting a single trajectory.
        basis: optional 
        """

        print "Generating basis..."
        prefix_dict, suffix_dict = hankel.top_k_basis(data, max_basis_size)
        #prefix_dict, suffix_dict = fair_basis(data, max_basis_size, len(data[0]))

        print "Estimating hankels..."

        # Note: all matrices returned by construct_hankels are csr_matrices
        hankels = hankel.construct_hankels(
            data, prefix_dict, suffix_dict, self.observations)

        hp, hs, hankel_matrix, symbol_hankels = hankels
        print "HP", hp
        print "HS", hs

        self.hankel = hankel_matrix
        self.symbol_hankels = symbol_hankels

        n_components = min(n_components, hankel_matrix.shape[0])

        print "Performing SVD..."
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

        print "Computing operators..."
        for o in symbol_hankels:
            symbol_hankel = P_plus.dot(symbol_hankels[o])
            symbol_hankel = symbol_hankel.dot(S_plus)
            self.B_o[o] = symbol_hankel.toarray()

        # computing stopping and starting vectors

        # P b_inf = hp => b_inf = P^+ hp
        self.b_inf = P_plus.dot(hp)
        self.b_inf = self.b_inf.toarray()[:, 0]

        # See Lemma 6.1.1 in Borja's thesis
        # B = sum(self.B_o.values())
        # self.b_inf = np.linalg.pinv(np.eye(n_components)-B).dot(self.b_inf)

        # b_0 S = hs => b_0 = hs S^+
        self.b_0 = hs.dot(S_plus)
        self.b_0 = self.b_0.toarray()[0, :]

        self.reset()

        return self.b_0, self.B_o, self.b_inf

    def update(self, obs):
        """ Update state upon seeing an observation (i.e. do filtering) """
        B_o = self.B_o[obs]
        numer = self.b.dot(B_o)
        denom = numer.dot(self.b_inf)

        self.b = numer / denom

    def reset(self):
        self.b = self.b_0.copy()

    def get_prediction(self):
        """
        Return the symbol that the model expects next.
        """
        predict = lambda o: self.get_obs_prob(o)
        return max(self.observations, key=predict)

    def get_obs_prob(self, o):
        """
        Returns the probablilty of observing o given the current state.
        """
        prob = self.b.dot(self.B_o[o]).dot(self.b_inf)
        return np.clip(prob, np.finfo(float).eps, 1)

    def get_obs_rank(self, o):
        """
        Get the rank of the given observation, in terms of probability
        of being the next observation.
        """
        probs = np.array(
            [self.get_obs_prob(obs) for obs in self.observations])

        return (
            np.count_nonzero(probs > self.get_obs_prob(o)),
            np.count_nonzero(probs < self.get_obs_prob(o)))

    def get_seq_prob(self, seq):
        """Returns the probability of a sequence given current state"""

        state = self.b
        for o in seq:
            state = state.dot(self.B_o[o])

        prob = state.dot(self.b_inf)

        return np.clip(prob, np.finfo(float).eps, 1)

    def get_state_for_seq(self, seq, initial_state=None):
        """Returns the probability of a sequence given current state"""

        state = self.b.T if initial_state is None else initial_state

        for o in seq:
            state = state.dot(self.B_o[o])

        return state / state.dot(self.b_inf)

    def get_WER(self, test_data):
        """Returns word error rate for the test data"""
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
        """Returns average log likelihood for the test data"""
        llh = 0

        for seq in test_data:
            seq_llh = 0

            self.reset()

            for o in seq:
                if base == 2:
                    seq_llh += np.log2(self.get_obs_prob(o))
                else:
                    seq_llh += np.log(self.get_obs_prob(o))

                self.update(o)

            llh += seq_llh

        return llh / len(test_data)


class SpectralPSRWithActions(object):
    def __init__(
            self, actions, observations, max_dim=80):

        self.n_actions = len(actions)
        self.actions = actions

        self.n_observations = len(observations)
        self.observations = observations

        self.B_ao = {}
        self.max_dim = max_dim

    def fit(self, data, max_basis_size, n_components, use_naive=False):
        """
        data should be a list of lists. Each sublist correspodnds to a
        trajectory.  Each entry of the trajectory should be a 2-tuple,
        giving the action followed by the observation.
        """

        print "Generating basis..."
        prefix_dict, suffix_dict = hankel.top_k_basis(data, max_basis_size)

        print "Estimating hankels..."

        # Note: all matrices returned by construct_hankels are csr_matrices
        if use_naive:
            print "...using naive estimator..."
            hankels = hankel.construct_hankels_with_actions(
                data, prefix_dict, suffix_dict,
                self.actions, self.observations)

            hp, hs, hankel_matrix, symbol_hankels = hankels
        else:
            print "...using robust estimator..."
            hankels = hankel.construct_hankels_with_actions_robust(
                data, prefix_dict, suffix_dict,
                self.actions, self.observations)

            hp, hs, hankel_matrix, symbol_hankels = hankels

        self.hankel = hankel_matrix
        self.symbol_hankels = symbol_hankels

        n_components = min(n_components, hankel_matrix.shape[0])

        print "Performing SVD..."
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

        print "Computing operators..."

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
    classifier_cls: class
        A classifier class. Instances of this class need to have a method `fit`
        which takes arguments X giving samples (shape N x p, where p is # of
        features) and Y giving labels (shape N) (basically the sklearn
        interface).

    classifier_args: list
        Positional arguments to instantiate the classifier with.

    classifier_kwargs: dict
        Keyword arguments to instantiate the classifier with.

    max_basis_size: int
        The maximum size of the basis for the spectral learner.

    n_components: int
        The number of components to use in the spectral learner.
    """
    def __init__(
            self, classifier_cls, classifier_args=None,
            classifier_kwargs=None, max_basis_size=500, n_components=50):

        self.classifier_cls = classifier_cls

        if classifier_args is None:
            classifier_args = []
        self.classifier_args = classifier_args

        if classifier_kwargs is None:
            classifier_kwargs = {}
        self.classifier_kwargs = classifier_kwargs

        self.max_basis_size = max_basis_size
        self.n_components = n_components

    def fit(self, pomdp, trajectories, actions):
        """
        Returns a policy trained on the given data.

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

        self.b_0, self.B_ao, self.b_inf = self.psr.fit(
            trajectories, self.max_basis_size, self.n_components)

        psr_states = []
        flat_actions = []

        for t, response_actions in zip(trajectories, actions):
            self.psr.reset()

            for (a, o), response_action in zip(t, response_actions):
                psr_states.append(self.psr.b)
                flat_actions.append(response_action)

                self.psr.update(a, o)

        classifier = self.classifier_cls(
            *self.classifier_args, **self.classifier_kwargs)

        # Most sklearn predictors operate on strings or numbers
        action_lookup = {str(a): a for a in set(flat_actions)}
        str_actions = [str(a) for a in flat_actions]

        classifier.fit(psr_states, str_actions)

        def f(psr_state):
            action_string = classifier.predict(psr_state)[0]
            return action_lookup[action_string]

        return SpectralPolicy(self.psr, f)


class SpectralPolicy(POMDPPolicy):
    def __init__(self, psr, f):
        self.psr = psr
        self.f = f

    def reset(self, init_dist=None):
        if init_dist is not None:
            raise Exception(
                "Cannot supply initiazation distribution to PSR. "
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
    for i in xrange(n_trajectories):
        trajectory = pomdp.sample_trajectory(
            exploration_policy, horizon, reset=True,
            return_reward=False, display=False)

        trajectories.append(trajectory)

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

                actual_obs, _ = pomdp.execute_action(action)

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
        for i in xrange(n_test_trajectories):
            trajectory = pomdp.sample_trajectory(
                exploration_policy, horizon, reset=True,
                return_reward=False, display=False)

            test_trajectories.append(trajectory)

        print "Word error rate: ", psr.get_WER(test_trajectories)

        llh = psr.get_log_likelihood(test_trajectories, base=2)
        print "Average log likelihood: ", llh
        print "Perplexity: ", 2**(-llh)
