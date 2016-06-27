import numpy as np

from spectral_dagger.sequence import HMM
from spectral_dagger.utils import indent


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
        super(MarkovChain, self).__init__(
            init_dist, T, np.eye(T.shape[0]), stop_prob)

    def __repr__(self):
        return str(self)

    def __str__(self):
        s = "MarkovChain\n"
        s += indent("n_states/n_symbols: %d\n" % self.n_states, 1)
        s += indent("init_dist:\n", 1)
        s += indent(str(self.init_dist) + "\n", 2)
        s += indent("T:\n", 1)
        s += indent(str(self.T) + "\n", 2)
        s += indent("stop:\n", 1)
        s += indent(str(self.stop_prob) + "\n", 2)
        return s
