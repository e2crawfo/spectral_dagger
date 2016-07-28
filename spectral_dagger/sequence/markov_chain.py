import numpy as np

from spectral_dagger.sequence import HMM
from spectral_dagger.utils import indent, normalize


class MarkovChain(HMM):

    def __init__(self, init_dist, T, stop_prob=None):
        """ A discrete Markov Chain.

        Convention: We assume a MarkovChain is just an HMM with an observation
        operator equal to the identity matrix. Consequently, it is possible for
        a MarkovChain to emit a sequence of length 0; if it is determined that
        we are going to halt in the initial state, then no transition takes
        place and no observation is emitted.

        Parameters
        ----------
        init_dist: ndarray
            A |states| vector specifying the initial state distribution.
            Defaults to a uniform distribution.
        T: ndarray
            A |states| x |states| matrix. Entry (i, j) gives
            the probability of moving from state i to state j, so each row of
            T must be a probability distribution.
        stop_prob: ndarray
            A |states| vector specifying the prob of halting in each state.

        """
        T = np.array(T)
        T[np.isclose(T.sum(1), 0), :] = np.ones(T.shape[1]) / T.shape[1]
        super(MarkovChain, self).__init__(
            init_dist, T, np.eye(T.shape[0]), stop_prob)

    def __repr__(self):
        return str(self)

    def __str__(self):
        s = "MarkovChain(\n"
        s += indent("n_states/n_symbols: %d\n" % self.n_states, 1)
        s += indent("init_dist:\n", 1)
        s += indent(str(self.init_dist) + "\n", 2)
        s += indent("T:\n", 1)
        s += indent(str(self.T) + "\n", 2)
        s += indent("stop:\n", 1)
        s += indent(str(self.stop_prob) + "\n", 2)
        s += ")"
        return s

    @property
    def n_observations(self):
        return self.n_states


class AdjustedMarkovChain(MarkovChain):

    def __init__(self, init_dist, T, stop_prob=None):
        """ An adjusted discrete Markov Chain.

        Adjusted so that it can never output empty sequences.

        """
        self.markov_chain = markov_chain = MarkovChain(init_dist, T, stop_prob)

        init_dist, T, stop_prob = (
            markov_chain.init_dist, markov_chain.T, markov_chain.stop_prob)

        # Create an alternate set of parameters such that it is not
        # possible to halt from the initial state. Effectively creates a
        # new state, and this state is the only state in which halting
        # can take place. The old halting probabilities becomes the
        # probabilities of transitioning into the halting state.
        n_symbols = len(init_dist)
        new_init_dist = np.array(list(init_dist) + [0.0])
        new_stop_prob = np.array([0.0] * n_symbols + [1.0])
        T = np.diag(1 - stop_prob).dot(T)
        new_T = np.concatenate((T, stop_prob.reshape(-1, 1)), axis=1)
        new_bottom_row = np.array([0.0] * n_symbols + [1.0]).reshape(1, -1)
        new_T = np.concatenate((new_T, new_bottom_row), axis=0)

        super(MarkovChain, self).__init__(
            new_init_dist, new_T, np.eye(n_symbols + 1), new_stop_prob)

    def __repr__(self):
        return str(self)

    def __str__(self):
        s = "AdjustedMarkovChain\n"
        s += indent("n_states/n_symbols: %d\n" % self.n_states, 1)
        s += indent("init_dist:\n", 1)
        s += indent(str(self.init_dist) + "\n", 2)
        s += indent("T:\n", 1)
        s += indent(str(self.T) + "\n", 2)
        s += indent("stop:\n", 1)
        s += indent(str(self.stop_prob) + "\n", 2)
        s += indent("Original Markov Chain: \n", 1)
        s += indent(str(self.markov_chain), 2)
        return s

    @property
    def init_dist(self):
        return self.markov_chain.init_dist

    @property
    def T(self):
        return self.markov_chain.T

    @property
    def stop_prob(self):
        return self.markov_chain.stop_prob

    @property
    def n_states(self):
        return self.markov_chain.n_states

    @staticmethod
    def from_sequences(sequences, learn_halt, n_symbols=None):
        """ If ``learn_halt`` is true, data must be generated
            from an "adjusted" Markov Chain. """
        if n_symbols is None:
            n_symbols = max(s for seq in sequences for s in seq) + 1

        pi = np.zeros(n_symbols)
        T = np.zeros((n_symbols, n_symbols))
        halt = np.zeros(n_symbols)

        for seq in sequences:
            pi[seq[0]] += 1.0

            for i in range(len(seq) - 1):
                T[seq[i], seq[i+1]] += 1.0

            if learn_halt:
                halt[seq[-1]] += 1.0

        pi = normalize(pi, ord=1)
        if learn_halt:
            halt /= T.sum(1) + halt
        else:
            learn_halt = np.zeros(n_symbols)
        T = normalize(T, ord=1, axis=1)

        return AdjustedMarkovChain(pi, T, halt)

    @staticmethod
    def from_distribution(distribution, sequences, learn_halt, n_symbols=None):
        """ If ``learn_halt`` is true, data must be generated
            from an "adjusted" Markov Chain. """
        if n_symbols is None:
            n_symbols = max(s for seq in sequences for s in seq) + 1

        pi = np.zeros(n_symbols)
        T = np.zeros((n_symbols, n_symbols))
        halt = np.zeros(n_symbols)

        for d, seq in zip(distribution, sequences):
            pi[seq[0]] += d

            for i in range(len(seq) - 1):
                T[seq[i], seq[i+1]] += d

            if learn_halt:
                halt[seq[-1]] += d

        pi = normalize(pi, ord=1)
        if learn_halt:
            halt /= T.sum(1) + halt
        else:
            learn_halt = np.zeros(n_symbols)
        T = normalize(T, ord=1, axis=1)

        return AdjustedMarkovChain(pi, T, halt)
