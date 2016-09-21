import numpy as np
from sklearn.utils import check_random_state
from scipy.misc import logsumexp
from scipy.stats import multivariate_normal

from spectral_dagger import Space, Environment
from spectral_dagger.utils.math import sample_multinomial, normalize
from spectral_dagger.utils.matlab import run_matlab_code

machine_eps = np.finfo(float).eps


class ContinuousHMM(Environment):

    def __init__(self, init_dist, T, O, states=None):
        """ A Hidden Markov Model with continuous observations.

        Parameters
        ----------
        init_dist: ndarray
            A |states| vector specifying the initial state distribution.
            Defaults to a uniform distribution.
        T: ndarray
            A |states| x |states| matrix. Entry (i, j) gives
            the probability of moving from state i to state j, so each row of
            T must be a probability distribution.
        O: A list of distributions.
            A list of length |states|. The i-th element is a distribution (so
            has an ``rvs`` method, which takes an argument called
            ``random_state``, and a ``pdf`` method; basically, instances
            of scipy.stats.rv_frozen) which gives the probability density
            for observation emissions given the i-th state. All distributions
            should have the same dimensionality.
        states: iterable (optional)
            The state space of the HMM.

        """
        self._states = range(T.shape[0]) if states is None else states
        n_states = len(self._states)

        self.init_dist = init_dist.copy()
        assert(self.init_dist.size == n_states)
        assert(np.allclose(sum(init_dist), 1.0))

        self._T = T.copy()
        self._T.flags.writeable = False

        assert np.allclose(np.sum(self._T, axis=1), 1.0)
        assert np.all(self._T >= 0) and np.all(self._T <= 1)
        assert self._T.shape == (n_states, n_states)

        self._O = O

        assert len(O) == n_states
        for o in O:
            assert hasattr(o, 'rvs')
            assert hasattr(o, 'pdf')

        self.reset()

    @property
    def name(self):
        return "ContinuousHMM"

    @property
    def action_space(self):
        return None

    @property
    def observation_space(self):
        return Space([-np.inf, np.inf] * len(self._O), "ObsSpace")

    def has_reward(self):
        return False

    def has_terminal_states(self):
        return False

    def in_terminal_state(self):
        return False

    @property
    def n_states(self):
        return len(self._states)

    @property
    def obs_dim(self):
        return self._O[0].rvs().size

    def reset(self, initial=None):
        self._state = self._states[
            sample_multinomial(self.init_dist, self.random_state)]

    def step(self):
        """ Returns the resulting observation. """
        o = self._O[self._state].rvs(random_state=self.random_state)
        self._state = self._states[
            sample_multinomial(self._T[self._state], self.random_state)]

        return o

    def cond_obs_prob(self, o):
        """ Get probability of observing o given the current state. """
        return self._O[self._state].pdf(o)

    def cond_obs_probs(self, o):
        """ Get vector containing probs of observing obs given each state. """
        return np.array(
            [self._O[s].pdf(o) for s in self._states])


class GMM(object):
    def __init__(self, pi, means, covs, careful=True):
        assert np.isclose(sum(pi), 1)
        self.pi = pi
        self.logpi = np.log(pi)
        self.dists = [multivariate_normal(mean=m, cov=c) for m, c in zip(means, covs)]
        self.means = means
        self.covs = covs
        self.careful = careful

    def pdf(self, o):
        if self.careful:
            return np.exp(self.logpdf(o))
        else:
            return self.pi.dot(np.array([d.pdf(o) for d in self.dists]))

    def logpdf(self, o):
        if self.careful:
            return logsumexp([lpi + d.logpdf(o) for lpi, d in zip(self.logpi, self.dists)])
        else:
            return np.log(self.pdf(o))

    def largest_mode(self):
        return self.means[np.argmax(self.pi)]


class GmmHmm(object):
    seq_length = None

    def __init__(
            self, n_states, n_components, n_dim,
            max_iter=10, thresh=1e-4, verbose=1, cov_type='full',
            directory=".", random_state=None, careful=True):

        self.rng = check_random_state(random_state)
        self.n_states = n_states
        self.n_components = n_components
        self.n_dim = n_dim

        self.max_iter = max_iter
        self.thresh = thresh
        self.verbose = verbose
        self.cov_type = cov_type
        self.careful = careful

        self.has_fit = False

        self.directory = directory

    def _random_init(self):
        n_states = self.n_states
        n_components = self.n_components
        n_dim = self.n_dim

        prior = np.abs(self.rng.randn(n_states))
        prior = normalize(prior, ord=1)

        transmat = np.abs(self.rng.randn(n_states, n_states))
        transmat = normalize(transmat, ord=1, axis=1)

        mu = np.abs(self.rng.randn(n_dim, n_states, n_components))

        Sigma = np.zeros((n_dim, n_dim, n_states, n_components))
        for i in range(n_states):
            for j in range(n_components):
                if self.cov_type == 'full':
                    A = self.rng.rand(n_dim, n_dim)
                    Sigma[:, :, i, j] = n_dim * np.eye(n_dim) + 0.5 * (A + A.T)
                elif self.cov_type == 'diagonal':
                    Sigma[:, :, i, j] = np.diag(self.rng.rand(n_dim))
                elif self.cov_type == 'spherical':
                    Sigma[:, :, i, j] = self.rng.rand() * np.eye(n_dim)
                else:
                    raise Exception("Unrecognized covariance type %s." % self.cov_type)

        mixmat = np.abs(self.rng.randn(n_states, n_components))
        mixmat = normalize(mixmat, ord=1, axis=1)

        self.prior = prior
        self.transmat = transmat
        self.mu = mu
        self.Sigma = Sigma
        self.mixmat = mixmat

    @property
    def name(self):
        return "GmmHmm"

    @property
    def action_space(self):
        return None

    @property
    def observation_space(self):
        return Space([-np.inf, np.inf] * len(self._O), "ObsSpace")

    def has_reward(self):
        return False

    def has_terminal_states(self):
        return False

    def in_terminal_state(self):
        return False

    @property
    def obs_dim(self):
        return self.n_dim

    def fit(self, data, reuse=False):
        if not reuse or not self.has_fit:
            self._random_init()

        # TODO: Remove requirement that all sequences be the same length.
        data = self._pad_data(data)
        self.seq_length = len(data[0])

        prior = self.prior
        transmat = self.transmat
        mu = self.mu
        Sigma = self.Sigma
        mixmat = self.mixmat

        #TODO: won't work with empty sequences
        assert len(data) > 0
        assert np.array(data[0]).ndim == 2

        obj_array = np.zeros(len(data), dtype=np.object)
        for i, seq in enumerate(data):
            # Matlab wants the sequences with shape (n_dim, seq_length)
            obj_array[i] = np.array(seq).T

        assert prior.ndim == 1
        assert transmat.ndim == 2
        assert mu.ndim == 3
        assert Sigma.ndim == 4
        assert mixmat.ndim == 2

        matlab_kwargs = dict(
            data=obj_array, prior0=prior, transmat0=transmat, mu0=mu,
            Sigma0=Sigma, mixmat0=mixmat)

        matlab_code = "run_mhmm('{{infile}}', '{{outfile}}', {max_iter}, {thresh}, {verbose}, '{cov_type}'); exit;"
        matlab_code = matlab_code.format(
            max_iter=self.max_iter, thresh=self.thresh,
            verbose=int(self.verbose), cov_type=self.cov_type)
        results = run_matlab_code(matlab_code, working_dir=self.directory, **matlab_kwargs)

        self.prior = results['prior'].squeeze()
        self.transmat = results['transmat']

        self.mu = results['mu']
        if self.mu.ndim < 3:
            self.mu = self.mu[:, :, None]

        self.Sigma = results['Sigma']
        if self.Sigma.ndim < 4:
            self.Sigma = self.Sigma[:, :, :, None]

        self.mixmat = results['mixmat']
        if self.mixmat.ndim < 2:
            self.mixmat = self.mixmat[:, None]

        assert self.prior.ndim == 1
        assert self.transmat.ndim == 2
        assert self.mu.ndim == 3
        assert self.Sigma.ndim == 4
        assert self.mixmat.ndim == 2

        self.log_transmat = np.log(self.transmat)

        self.dists = [
            GMM(
                pi=self.mixmat[i, :],
                means=[self.mu[:, i, j] for j in range(self.n_components)],
                covs=[self.Sigma[:, :, i, j] for j in range(self.n_components)],
                careful=self.careful)
            for i in range(self.n_states)]
        self.reset()
        self.has_fit = True

        return self

    def _pad_data(self, data):
        assert len(data) > 0
        seq_length = len(data[0])
        if all([len(l) == seq_length for l in data[1:]]):
            return data

        max_len = max(len(d) for d in data)
        result = []
        for d in data:
            short_by = max_len-d.shape[0]
            d = np.concatenate((d, np.zeros((short_by, d.shape[1]))), axis=0)
            result.append(d)

        return result

    def _all_cond_log_likelihood(self, data):
        # TODO: Remove requirement that all sequences be the same length.
        data = self._pad_data(data)

        prior = self.prior
        transmat = self.transmat
        mu = self.mu
        Sigma = self.Sigma
        mixmat = self.mixmat

        #TODO: won't work with empty sequences
        assert len(data) > 0
        assert np.array(data[0]).ndim == 2

        obj_array = np.zeros(len(data), dtype=np.object)
        for i, seq in enumerate(data):
            # Matlab wants the sequences with shape (n_dim, seq_length)
            obj_array[i] = np.array(seq).T

        matlab_kwargs = dict(
            data=obj_array, prior=prior, transmat=transmat, mu=mu,
            Sigma=Sigma, mixmat=mixmat)

        matlab_code = "run_mhmm_cond_logprob('{infile}', '{outfile}');"
        results = run_matlab_code(matlab_code, working_dir=self.directory, **matlab_kwargs)

        # shape : (n_seqs, seq_length)
        cond_log_likelihood = results['cond_log_likelihood']
        return cond_log_likelihood

    def reset(self, initial=None):
        if self.careful:
            self._b = np.log(self.prior)
        else:
            self._b = self.prior

    def update(self, o):
        if self.careful:
            log_obs_prob = np.array([d.logpdf(o) for d in self.dists])
            v = self._b.reshape(-1, 1) + self.log_transmat
            self._b = log_obs_prob + logsumexp(v, axis=0)
            self._b -= logsumexp(self._b)
        else:
            self._b = self._b.dot(self.transmat) * np.array([d.pdf(o) for d in self.dists])
            self._b = normalize(self._b, ord=1)

    def cond_obs_prob(self, o, log=False):
        """ Get probability of observing o given the current state. """
        if self.careful:
            logprob = logsumexp(self._b + np.array([d.logpdf(o) for d in self.dists]))
            return logprob if log else np.exp(logprob)
        else:
            prob = self._b.dot(np.array([d.pdf(o) for d in self.dists]))
            return np.log(prob) if log else prob

    def cond_predict(self):
        most_likely = np.argmax(self._b)
        return self.dists[most_likely].largest_mode()

    def cond_predicts(self, seq):
        self.reset()
        predictions = []
        for o in seq:
            predictions.append(self.cond_predict())
            self.update(o)
        return np.array(predictions)

    def string_prob(self, string, log=False):
        self.reset()

        log_prob = 0.0

        for o in string:
            log_prob += self.cond_obs_prob(o, log=True)
            self.update(o)

        if log:
            return log_prob
        else:
            return np.exp(log_prob)

    def mean_log_likelihood(self, test_data, string=True):
        """ Get average log likelihood for the test data. """
        if len(test_data) == 0:
            return -np.inf

        llh = 0.0

        for seq in test_data:
            if string:
                seq_llh = self.string_prob(seq, log=True)
            else:
                seq_llh = self.prefix_prob(seq, log=True)

            llh += seq_llh

        return llh / len(test_data)

    def RMSE(self, test_data):
        n_predictions = sum(len(seq) for seq in test_data)
        if n_predictions == 0:
            return np.inf

        se = 0.0
        for seq in test_data:
            predictions = self.cond_predicts(seq)
            se += ((np.array(predictions) - np.array(seq))**2).sum()

        return np.sqrt(se / n_predictions)

if __name__ == "__main__":
    from spectral_dagger.datasets import pendigits

    data, _ = pendigits.get_data(difference=True, use_digits=[0])
    data = data[0]

    n_states = 20
    n_components = 1
    n_dim = 2

    gmm_hmm = GmmHmm(
        n_states, n_components, n_dim,
        max_iter=30, thresh=1e-4, verbose=1, cov_type='full',
        directory="temp", random_state=None, careful=False)

    gmm_hmm.fit(data)
    # clls = gmm_hmm._all_cond_log_likelihood(data)

    # seq_probs = clls.sum(axis=1)
    # print(seq_probs)
    # print(clls.sum() / len(data))

    print gmm_hmm.mean_log_likelihood(data)
    print gmm_hmm.RMSE(data)
