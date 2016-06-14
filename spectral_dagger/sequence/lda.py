import numpy as np
import logging
import gensim
from itertools import product
import time
from collections import defaultdict

from spectral_dagger.sequence import PredictiveStateRep, SpectralPSR
from spectral_dagger.utils import normalize


logger = logging.getLogger(__name__)

machine_eps = np.finfo(float).eps
MAX_BASIS_SIZE = 100


class LatentDirichletPSR(PredictiveStateRep):
    def __init__(self, observations):
        self.b_0 = None
        self.b_inf = None
        self.B_o = None

        self.observations = observations
        self.n_observations = len(self.observations)

    @staticmethod
    def _make_conditional_counts(samples, symbols):
        count_vectors = defaultdict(lambda: np.zeros(len(symbols)))

        for sample in samples:
            sample = tuple(sample)
            count_vectors[()][sample[0]] += 1
            count_vectors[sample[0:1]][sample[1]] += 1
            count_vectors[sample[0:2]][sample[2]] += 1

        return count_vectors

    @staticmethod
    def _conditional_to_joint(conditional_hankel, prefixes, symbols):
        """ Convert a conditional hankel to a joint hankel.

        Assumes the conditional hankel has a particular form.

        """
        n_symbols = len(symbols)

        h = np.zeros((n_symbols, n_symbols))

        sh = {s: np.zeros((n_symbols, n_symbols)) for s in symbols}

        prefix2idx = {p: i for i, p in enumerate(prefixes)}

        hp = conditional_hankel[prefix2idx[()]]
        hs = conditional_hankel[prefix2idx[()]]

        for symbol in symbols:
            h[symbol, :] = (
                hp[symbol] * conditional_hankel[prefix2idx[(symbol,)]])

            for symbol_prime in symbols:
                sh[symbol_prime][symbol] = (
                    h[symbol, symbol_prime] *
                    conditional_hankel[prefix2idx[(symbol, symbol_prime)]])
        return hp, hs, h, sh

    def fit(self, data, n_components, alpha=1.0, eta=1.0, direct=True):
        """ Fit a PSR to the given data using an LDA algorithm.

        Parameters
        ----------
        data: list of list of observations
            Each sublist is a list of observations constituting a trajectory.
        n_components: int
            Number of dimensions in feature space.
        alpha: float > 0
            Concentration parameter for prior on state distributions.
        eta: float > 0
            Concentration parameter for prior on observation distributions.
        direct: bool
            If True, use the factors obtained through LDA to do spectral
            learning. If False, construct a joint Hankel using the factors
            obtained through LDA, and perform spectral learning using that.

        """
        symbols = self.observations
        n_symbols = len(symbols)

        cc = self._make_conditional_counts(data, symbols)
        prefixes = (
            [()] + list(product(symbols)) + list(product(symbols, symbols)))

        bow = np.array([cc[p] for p in prefixes])
        bow = gensim.matutils.Dense2Corpus(bow, False)
        id2word = {i: str(seq) for i, seq in enumerate(symbols)}

        logger.info("Starting LDA")
        t0 = time.time()
        lda = gensim.models.ldamodel.LdaModel(
            corpus=bow, id2word=id2word,
            num_topics=n_components, update_every=0,
            passes=20, alpha=alpha, eta=eta)
        logger.info("Done LDA")
        logger.info("LDA Time: %f seconds", time.time() - t0)

        S = np.array([
            lda.state.get_lambda()[i] for i in range(n_components)]).T
        S = normalize(S, ord=1, axis=0).T

        n_docs = len(cc)
        documents = [lda[b] for b in bow]
        P = np.zeros((n_docs, n_components))
        for t, doc in enumerate(documents):
            for idx, prob in doc:
                P[t, idx] = prob

        if direct:
            conditional_hankel = P.dot(S)
            hp, _, _, sh = self._conditional_to_joint(
                conditional_hankel, prefixes, symbols)

            P = P[1:n_symbols+1, :]
            P_plus = np.linalg.pinv(P)
            S_plus = np.linalg.pinv(S)

            self.b_0 = hp.dot(S_plus)
            self.b_inf = P_plus.dot(hp)

            self.B_o = {
                o: P_plus.dot(sh[o]).dot(S_plus)
                for o in symbols}

        else:
            basis = {(symbol,): i for i, symbol in enumerate(symbols)}
            basis = basis, basis

            # reconstruction
            conditional_hankel = P.dot(S)
            hankels = self._conditional_to_joint(
                conditional_hankel, prefixes, symbols)
            psr = SpectralPSR(symbols)
            psr.fit(
                [], n_components, basis=basis,
                hankels=hankels, sparse=False)

            self.b_0, self.b_inf, self.B_o = psr.b_0, psr.b_inf, psr.B_o

        self.B = sum(self.B_o.values())
        self.compute_start_end_vectors(
            self.b_0, self.b_inf, estimator='prefix')

        self.reset()
