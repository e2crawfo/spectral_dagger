import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from sklearn.utils.extmath import randomized_svd
import heapq
from collections import defaultdict
import itertools

from policy import POMDPPolicy

VERBOSE = True


class SpectralPSRWithActions(object):
    def __init__(
            self, actions, observations, max_dim=80):

        self.num_actions = len(actions)
        self.actions = actions

        self.num_observations = len(observations)
        self.observations = observations

        self.buildtime = 0
        self.B_ao = {}
        self.max_dim = max_dim

    def fit(self, data, max_basis_size, num_components, use_naive=False):
        """
        data should be a list of lists. Each sublist correspodnds to a
        trajectory.  Each entry of the trajectory should be a 2-tuple,
        giving the action and the observation.
        """

        print "Generating basis..."
        prefix_dict, suffix_dict = top_k_basis(data, max_basis_size)

        print "Estimating hankels..."

        # Note: all matrices returned by construct_hankels are csr_matrices
        if use_naive:
            print "...using naive estimator..."
            hankels = construct_hankels_with_actions(
                data, prefix_dict, suffix_dict,
                self.actions, self.observations)

            hp, hs, hankel_matrix, symbol_hankels = hankels
        else:
            print "...using robust estimator..."
            hankels = construct_hankels_with_actions_robust(
                data, prefix_dict, suffix_dict,
                self.actions, self.observations)

            hp, hs, hankel_matrix, symbol_hankels = hankels

        self.hankel = hankel_matrix
        self.symbol_hankels = symbol_hankels

        num_components = min(num_components, hankel_matrix.shape[0])

        print "Performing SVD..."
        n_oversamples = 10
        n_iter = 5

        # H = U S V^T
        U, S, VT = randomized_svd(
            hankel_matrix, self.max_dim, n_oversamples, n_iter)

        V = VT.T

        U = U[:, :num_components]
        V = V[:, :num_components]
        S = np.diag(S[:num_components])

        # P^+ = (HV)^+ = (US)^+ = S^+ U+ = S^-1 U.T
        P_plus = csr_matrix((np.linalg.inv(S)).dot(U.T))

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
        self.b_inf = np.linalg.inv(np.eye(num_components)-B).dot(self.b_inf)

        # b_0 S = hs => b_0 = hs S^+
        self.b_0 = hs.dot(S_plus)
        self.b_0 = self.b_0.toarray()[0, :]

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

        return np.clip(prob, np.finfo(float).eps)

    def get_state_for_seq(self, seq, initial_state=None):
        """Returns the probability of a sequence given current state"""

        state = self.b.T if initial_state is None else initial_state

        for (a, o) in seq:
            state = state.dot(self.B_ao[(a, o)])

        return state / state.dot(self.b_inf)

    def get_WER(self, test_data):
        """Returns word error rate for the test data"""
        errors = 0
        num_predictions = 0

        for seq in test_data:
            self.reset()

            for a, o in seq:
                prediction = self.get_prediction(a)

                if prediction != o:
                    errors += 1

                self.update(a, o)

                num_predictions += 1

        return errors/float(num_predictions)

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


class PureSpectralLearningAlgorithm(object):
    def __init__(self):
        pass

    def fit(self, data):
        """
        Return a policy, trained on the data in 'data'. 'data' is a dictionary
        mapping from trajectories to lists of expert responses.
        """
        return SpectralPolicy()


class SpectralPlusClassifier(POMDPPolicy):
    def __init__(
            self, actions, observations, data):
        """
        data: list
          a list of pairs. pair[0] is a trajectory (sequence of
          action-observation pairs) and pair[1] is a sequence of expert
          actions. pair[1] can also be None, in which case it is an
          unlabelled data point.
        """
        self.psr = SpectralPSRWithActions(actions, observations)

        max_basis_size = 500
        num_components = 50

        trajectories = [d[0] for d in data]

        self.b_0, self.B_ao, self.b_inf = self.psr.fit(
            trajectories, max_basis_size, num_components)

        states = []
        expert_actions = []

        labelled_data = [d for d in data if d[1] is not None]

        for seq, ea_seq in labelled_data:
            self.psr.reset()

            for e, (a, o) in zip(ea_seq, seq):
                states.append(self.psr.b)
                expert_actions.append(str(e))
                self.psr.update(a, o)

        states = np.array(states)
        expert_actions = np.array(expert_actions)

        from sklearn import svm

        self.classifier = svm.SVC()
        self.classifier.fit(states, expert_actions)

        self.action_lookup = {str(a): a for a in actions}

    def reset(self, init_dist=None):
        if init_dist is not None:
            raise Exception(
                "Cannot supply initiazation distribution to PSR. "
                "Only works with the initialization distribution "
                "on which it was trained.")

        self.psr.reset()

    def update(self, action, observation):
        self.last_action = action
        self.psr.update(action, observation)

    def get_action(self):
        action_string = self.classifier.predict(self.psr.b)
        return self.action_lookup[action_string[0]]


def construct_hankels_with_actions(
        data, prefix_dict, suffix_dict, actions,
        observations, basis_length=100):

    size_P = len(prefix_dict)
    size_S = len(suffix_dict)

    hankel = lil_matrix((size_P, size_S))

    symbol_hankels = {}

    for pair in itertools.product(actions, observations):
        symbol_hankels[pair] = lil_matrix((size_P, size_S))

    action_sequence_counts = defaultdict(int)

    for seq in data:
        action_seq = tuple([pair[0] for pair in seq])
        action_sequence_counts[action_seq] += 1

    for seq in data:
        action_seq = tuple([pair[0] for pair in seq])
        action_seq_count = action_sequence_counts[action_seq]

        # iterating over prefix start positions
        for i in range(len(seq)+1):
            if i > basis_length:
                break

            if len(seq) - i > basis_length:
                break

            prefix = tuple(seq[:i])

            if i == len(seq):
                suffix = ()
            else:
                suffix = tuple(seq[i:len(seq)])

            if prefix in prefix_dict and suffix in suffix_dict:
                hankel[
                    prefix_dict[prefix],
                    suffix_dict[suffix]] += 1.0 / action_seq_count

            if i < len(seq):
                a, o = seq[i]

                if i + 1 == len(seq):
                    suffix = ()
                else:
                    suffix = tuple(seq[i+1:len(seq)])

                if prefix in prefix_dict and suffix in suffix_dict:
                    symbol_hankel = symbol_hankels[(a, o)]
                    symbol_hankel[
                        prefix_dict[prefix],
                        suffix_dict[suffix]] += 1.0 / action_seq_count

    hankel = csr_matrix(hankel)

    for pair in itertools.product(actions, observations):
        symbol_hankels[pair] = csr_matrix(symbol_hankels[pair])

    return (
        hankel[:, 0], hankel[0, :], hankel, symbol_hankels)


def construct_hankels_with_actions_robust(
        data, prefix_dict, suffix_dict, actions,
        observations, basis_length=100):

    size_P = len(prefix_dict)
    size_S = len(suffix_dict)

    hankel = lil_matrix((size_P, size_S))

    symbol_hankels = {}

    for pair in itertools.product(actions, observations):
        symbol_hankels[pair] = lil_matrix((size_P, size_S))

    prefix_count = defaultdict(int)

    for seq in data:
        seq = tuple(seq)
        for i in range(len(seq)):
            prefix = seq[:i] + (seq[i][0],)
            prefix_count[prefix] += 1

            prefix = seq[:i+1]
            prefix_count[prefix] += 1

    for seq in data:
        seq = tuple(seq)

        estimate = 1.0
        for i in range(len(seq)):
            numer = prefix_count[seq[:i+1]]
            denom = prefix_count[seq[:i] + (seq[i][0],)]
            estimate *= float(numer) / denom

        # iterating over prefix start positions
        for i in range(len(seq)+1):
            if i > basis_length:
                break

            if len(seq) - i > basis_length:
                break

            prefix = tuple(seq[:i])

            if i == len(seq):
                suffix = ()
            else:
                suffix = tuple(seq[i:len(seq)])

            if prefix in prefix_dict and suffix in suffix_dict:
                hankel[
                    prefix_dict[prefix],
                    suffix_dict[suffix]] = estimate

            if i < len(seq):
                if i + 1 == len(seq):
                    suffix = ()
                else:
                    suffix = tuple(seq[i+1:len(seq)])

                if prefix in prefix_dict and suffix in suffix_dict:
                    symbol_hankel = symbol_hankels[seq[i]]
                    symbol_hankel[
                        prefix_dict[prefix],
                        suffix_dict[suffix]] = estimate

    hankel = csr_matrix(hankel)

    for pair in itertools.product(actions, observations):
        symbol_hankels[pair] = csr_matrix(symbol_hankels[pair])

    return (
        hankel[:, 0], hankel[0, :], hankel, symbol_hankels)


def top_k_basis(data, k):
    prefix_count_dict = defaultdict(int)
    suffix_count_dict = defaultdict(int)

    for seq in data:
        for i in range(0, len(seq)+1):
            prefix = tuple(seq[:i])

            if prefix:
                prefix_count_dict[prefix] += 1

            if i < len(seq):
                suffix = tuple(seq[i:len(seq)])

                if suffix:
                    suffix_count_dict[suffix] += 1

    top_k_prefix = [()]
    top_k_prefix.extend(
        heapq.nlargest(k, prefix_count_dict, key=prefix_count_dict.get))

    top_k_suffix = [()]
    top_k_suffix.extend(
        heapq.nlargest(k, suffix_count_dict, key=suffix_count_dict.get))

    min_length = min(len(top_k_prefix), len(top_k_suffix))
    top_k_prefix = top_k_prefix[:min_length]
    top_k_suffix = top_k_suffix[:min_length]

    prefix_dict = {item: index for (index, item) in enumerate(top_k_prefix)}
    suffix_dict = {item: index for (index, item) in enumerate(top_k_suffix)}

    return prefix_dict, suffix_dict


def fair_basis(data, k, horizon):
    """Returns a basis with the top k elements at each length.
    Currently assumes all trajectories are the same length."""

    prefixes = set(())
    suffixes = set(())

    for i in range(0, horizon+1):
        prefix_count_dict = defaultdict(int)
        suffix_count_dict = defaultdict(int)

        for seq in data:
            prefix = tuple(seq[:i])

            if prefix:
                prefix_count_dict[prefix] += 1

            if i < len(seq):
                suffix = tuple(seq[i:horizon])

                if suffix:
                    suffix_count_dict[suffix] += 1

        best_prefixes = heapq.nlargest(
            k, prefix_count_dict, key=prefix_count_dict.get)

        for p in best_prefixes:
            prefixes.add(p)

        best_suffixes = heapq.nlargest(
            k, suffix_count_dict, key=suffix_count_dict.get)

        for s in best_suffixes:
            suffixes.add(s)

    while len(prefixes) < len(suffixes):
        t = np.random.randint(len(data))
        i = np.random.randint(horizon)

        prefixes.add(tuple(data[t][:i+1]))

    while len(suffixes) < len(prefixes):
        t = np.random.randint(len(data))
        i = np.random.randint(horizon)

        prefixes.add(tuple(data[t][i:]))

    prefixes = list(prefixes)
    suffixes = list(suffixes)

    prefix_dict = {item: index for (index, item) in enumerate(prefixes)}
    suffix_dict = {item: index for (index, item) in enumerate(suffixes)}

    return prefix_dict, suffix_dict

if __name__ == "__main__":
    import grid_world
    from policy import POMDPPolicy

    # Sample a bunch of trajectories, run the learning algorithm on them
    num_trajectories = 20000
    horizon = 3
    num_components = 40
    max_basis_size = 1000
    max_dim = 150

    world = np.array([
        ['x', 'x', 'x', 'x', 'x'],
        ['x', ' ', ' ', 'G', 'x'],
        ['x', ' ', ' ', ' ', 'x'],
        ['x', ' ', 'x', ' ', 'x'],
        ['x', ' ', ' ', ' ', 'x'],
        ['x', 'A', ' ', ' ', 'x'],
        ['x', 'x', 'x', 'x', 'x']]
    )

    num_colors = 2

    pomdp = grid_world.EgoGridWorld(num_colors, world)

    exploration_policy = POMDPPolicy()
    exploration_policy.fit(pomdp)

    trajectories = []

    print "Sampling trajectories..."
    for i in xrange(num_trajectories):
        trajectory = pomdp.sample_trajectory(
            exploration_policy, horizon, reset=True,
            return_reward=False, display=False)

        trajectories.append(trajectory)

    for use_naive in [True, False]:
        print "Training model..."

        psr = SpectralPSRWithActions(
            pomdp.actions, pomdp.observations, max_dim)

        b_0, B_ao, b_inf = psr.fit(
            trajectories, max_basis_size, num_components, use_naive)

        test_length = 10
        num_tests = 2000

        num_below = []
        top_three_count = 0

        print "Running tests..."

        display = False

        for t in range(num_tests):

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

                num_below.append(rank[1])
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
            "Average num below: ", np.mean(num_below),
            "of", len(pomdp.observations))
        print "Probability in top 3: %f" % (
            float(top_three_count) / (test_length * num_tests))

        num_test_trajectories = 40
        test_trajectories = []

        print "Sampling test trajectories for WER..."
        for i in xrange(num_test_trajectories):
            trajectory, reward = pomdp.sample_trajectory(
                exploration_policy, horizon, reset=True,
                return_reward=False, display=False)

            test_trajectories.append(trajectory)

        print "Word error rate: ", psr.get_WER(test_trajectories)

        llh = psr.get_log_likelihood(test_trajectories, base=2)
        print "Average log likelihood: ", llh
        print "Perplexity: ", 2**(-llh)
