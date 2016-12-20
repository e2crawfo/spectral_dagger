from rpy2 import robjects
from rpy2.robjects import r, numpy2ri
from sklearn.utils import check_random_state
import numpy as np

from spectral_dagger.sequence import MixtureSeqGen, HMM
from spectral_dagger.sequence.hmm import perturb_hmm, dummy_hmm


class MixtureHMM(MixtureSeqGen):
    def __init__(self, n_components, n_states, n_observations):
        """
        Parameters
        ----------
        n_components: int
            Number of mixture components.
        n_states: int or list
            Number of states for mixture components.
            If an integer provided, it is used for all components.
        n_observations: int
            Number of observations.

        """
        self.n_components = n_components
        self._n_observations = n_observations
        self.n_states = n_states
        self.coefficients = 1.0 / n_components * np.ones(n_components)
        self.seq_gens = [
            dummy_hmm(self.n_states, n_observations)
            for i in range(n_components)]

        # Create a separate set of SG's for sampling and filtering
        # since problems can arise if a single object is used for
        # both sampling and filtering simultaneously
        # TODO: Make this work.
        # self._filter_seq_gens = [
        #     seq_gen.deepcopy() for seq_gen in seq_gens]
        # self._sample_seq_gens = [
        #     seq_gen.deepcopy() for seq_gen in seq_gens]
        # self._filter_seq_gens = seq_gens
        # self._sample_seq_gens = seq_gens

        # self.reset()


    def __str__(self):
        s = "<MixtureHMM. coefficients: %s,\n" % self.coefficients

        for i, seq_gen in enumerate(self.seq_gens):
            s += ", %d: %s\n" % (i, seq_gen)
        s += ">"

        return s

    def fit(self, data, noise=0.1, initial_guess=None, rng=None):
        rng = check_random_state(rng)
        if initial_guess is None:
            base_hmm = dummy_hmm(self.n_states, self.n_observations)
            initial_guess = [
                perturb_hmm(base_hmm, noise, preserve_zeros=False, rng=rng)
                for i in range(self.n_components)]
            initial_guess = [hmm.modified() for hmm in initial_guess]

        r("library(TraMineR)")
        r("library(seqHMM)")

        max_length = max(len(seq) for seq in data)
        padded_data = np.zeros((len(data), max_length))
        padding_value = self.n_observations
        for i, seq in enumerate(data):
            padded_seq = seq + [padding_value] * (max_length - len(seq))
            padded_data[i, :] = padded_seq

        numpy2ri.activate()
        robjects.globalenv['data'] = r.matrix(padded_data, nrow=len(seq))

        r("sequences <- seqdef(data=data)")

        r("initial_probs = vector('list', {0})".format(self.n_components))
        r("emission_probs = vector('list', {0})".format(self.n_components))
        r("transition_probs = vector('list', {0})".format(self.n_components))

        for i, hmm in enumerate(initial_guess):
            assert isinstance(hmm, HMM)

            r['initial_probs'][i] = robjects.FloatVector(hmm.init_dist)
            r['emission_probs'][i] = r.matrix(hmm.O, nrow=hmm.O.shape[0])
            r['transition_probs'][i] = r.matrix(hmm.T, nrow=hmm.T.shape[0])
        numpy2ri.deactivate()

        r("init_mhmm <- build_mhmm("
          "    observations = list(sequences), "
          "    initial_probs = initial_probs, "
          "    emission_probs = emission_probs, "
          "    transition_probs = transition_probs "
          ")")
        r("mhmm_fit <- fit_model(init_mhmm)")

if __name__ == "__main__":
    from spectral_dagger.sequence import MixtureSeqGen
    from spectral_dagger.sequence.hmm import bernoulli_hmm
    from spectral_dagger.datasets.pautomac import make_pautomac_like
    rng = check_random_state(1)
    n_components = 2
    n_obs = 3
    n_states = 4
    o_density, t_density = 0.5, 0.2
    #hmms = [bernoulli_hmm(n_states, n_obs, rng) for i in range(n_components)]
    hmms = [
        make_pautomac_like(
            'hmm', n_states, n_obs, o_density, t_density, halts=True)
        for i in range(n_components)]

    hmms = [hmm.modified() for hmm in hmms]
    coefficients = rng.dirichlet(np.ones(n_components))

    print(hmms[0].sample_episodes(10, horizon=10))

    mixture = MixtureSeqGen(coefficients, hmms)

    # data = mixture.sample_episodes(1, horizon=10)
    # print(data)

    x = MixtureHMM(n_components, n_states, n_obs)
    x.fit(data)
