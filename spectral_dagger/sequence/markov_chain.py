import numpy as np

from spectral_dagger.sequence import HMM


class MarkovChain(HMM):

    def __init__(self, T, init_dist=None, stop_prob=None):
        """ A discrete Markov Chain.

        Convention: We assume a MarkovChain is just an HMM with an observation
        operator equal to the identity matrix. Consequently, it is possible for
        a MarkovChain to emit a sequence of length 0; if it is determined that
        we are going to halt in the initial state, then no transition takes
        place and no observation is emitted.

        Parameters
        ----------
        T: ndarray
            A |states| x |states| matrix. Entry (i, j) gives
            the probability of moving from state i to state j, so each row of
            T must be a probability distribution.
        init_dist: ndarray
            A |states| vector specifying the initial state distribution.
            Defaults to a uniform distribution.
        stop_prob: ndarray
            A |states| vector specifying the prob of halting in each state.

        """
        super(MarkovChain, self).__init__(
            T, np.eye(T.shape[0]), init_dist, stop_prob)

    def __str__(self):
        return "<MarkovChain. n_states: %d>" % self.n_states
