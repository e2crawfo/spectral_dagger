import numpy as np
import os
from sklearn.utils import check_random_state
from scipy.misc import logsumexp
import shutil

from spectral_dagger import Space, Environment, Estimator
from spectral_dagger.sequence import SequenceModel
from spectral_dagger.utils.math import sample_multinomial, normalize
from spectral_dagger.utils.dists import MixtureDist, GMM
from spectral_dagger.utils.matlab import run_matlab_code
from spectral_dagger.datasets import pendigits

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
        self._states = list(range(T.shape[0])) if states is None else states
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


class GmmHmm(SequenceModel, Estimator):
    """
    Parameters
    ----------
    n_states: int > 0
        Number of hidden state of the GmmHMM.
    n_components: int > 0
        Number of (Gaussian) mixture components for each state-conditional emission distribution.
    n_dim: int > 0
        Dimensionality of output.

    """
    has_fit = False

    def __init__(
            self, n_states=1, n_components=1, n_dim=1,
            max_iter=10, thresh=1e-4, verbose=1, cov_type='full',
            directory=".", random_state=None, careful=True,
            left_to_right=False, reuse=False, n_restarts=1,
            max_attempts=10, raise_errors=False, initial_params=None,
            delete_matlab_files=True):

        self.random_state = random_state
        self.n_states = n_states
        self.n_components = n_components
        self.n_dim = n_dim
        self.max_iter = max_iter
        self.thresh = thresh
        self.verbose = verbose
        self.cov_type = cov_type
        self.careful = careful
        self.left_to_right = left_to_right
        self.reuse = reuse
        self.n_restarts = n_restarts
        self.max_attempts = max_attempts
        self.raise_errors = raise_errors
        self.delete_matlab_files = delete_matlab_files
        self.directory = directory
        self.initial_params = initial_params

    @property
    def record_attrs(self):
        return super(GmmHmm, self).record_attrs or set(['n_states'])

    def point_distribution(self, context):
        pd = super(GmmHmm, self).point_distribution(context)
        if 'max_states' in context:
            pd.update(n_states=list(range(2, context['max_states'])))
        return pd

    def _validate_params(self):
        if self.mu.ndim < 3:
            self.mu = self.mu[:, :, None]

        if self.sigma.ndim < 4:
            self.sigma = self.sigma[:, :, :, None]

        if self.M.ndim < 2:
            self.M = self.M[:, None]

        assert self.pi.ndim == 1
        assert self.T.ndim == 2
        assert self.mu.ndim == 3
        assert self.sigma.ndim == 4
        assert self.M.ndim == 2

        assert self.pi.shape == (self.n_states,)
        assert self.T.shape == (self.n_states, self.n_states)
        assert self.mu.shape == (self.n_dim, self.n_states, self.n_components)
        assert self.sigma.shape == (self.n_dim, self.n_dim, self.n_states, self.n_components)
        assert self.M.shape == (self.n_states, self.n_components)

        assert np.isclose(self.pi.sum(), 1.0)

        if False:
            assert np.allclose(self.T.sum(1), 1.0)
        else:
            for t in self.T:
                if np.isclose(t.sum(), 0.0):
                    t[0] = 1.0

        self.log_T = np.log(self.T)
        self.dists = [
            GMM(
                pi=self.M[i, :],
                means=[self.mu[:, i, j] for j in range(self.n_components)],
                covs=[self.sigma[:, :, i, j] for j in range(self.n_components)])
            for i in range(self.n_states)]

    def _random_init(self, random_state):
        n_states = self.n_states
        n_components = self.n_components
        n_dim = self.n_dim

        pi = np.abs(self.random_state.randn(n_states))
        pi = normalize(pi, ord=1)

        T = np.abs(self.random_state.randn(n_states, n_states))
        if self.left_to_right:
            T = np.triu(T)
        T = normalize(T, ord=1, axis=1)

        mu = np.abs(self.random_state.randn(n_dim, n_states, n_components))

        sigma = np.zeros((n_dim, n_dim, n_states, n_components))
        for i in range(n_states):
            for j in range(n_components):
                if self.cov_type == 'full':
                    A = self.random_state.rand(n_dim, n_dim)
                    sigma[:, :, i, j] = n_dim * np.eye(n_dim) + 0.5 * (A + A.T)
                elif self.cov_type == 'diagonal':
                    sigma[:, :, i, j] = np.diag(self.random_state.rand(n_dim))
                elif self.cov_type == 'spherical':
                    sigma[:, :, i, j] = self.random_state.rand() * np.eye(n_dim)
                else:
                    raise Exception("Unrecognized covariance type %s." % self.cov_type)

        M = np.abs(self.random_state.randn(n_states, n_components))
        M = normalize(M, ord=1, axis=1)

        self.pi = pi
        self.T = T
        self.mu = mu
        self.sigma = sigma
        self.M = M
        self._validate_params()

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

    def fit(self, data, reuse=None):
        reuse = self.reuse if reuse is None else reuse
        print(("*" * 40))
        print(("Beginning new fit for %s with reuse=%r." % (self.__class__.__name__, reuse)))

        warm_start = self.initial_params or (self.has_fit and reuse)
        if self.initial_params and not (self.has_fit and reuse):
            self.pi = self.initial_params['pi'].copy()
            self.T = self.initial_params['T'].copy()
            self.mu = self.initial_params['mu'].copy()
            self.sigma = self.initial_params['sigma'].copy()
            self.M = self.initial_params['M'].copy()
            self._validate_params()

        n_restarts = 1 if warm_start else self.n_restarts
        n_iters = 0
        log_likelihood = 0
        results = []
        random_state = check_random_state(self.random_state)

        for i in range(n_restarts):
            print(("Beginning restart: %d" % i))
            n_attempts = 0

            while True:
                n_attempts += 1
                if not warm_start:
                    self._random_init(random_state)

                # TODO: Remove requirement that all sequences be the same length.
                data = self._pad_data(data)
                self.seq_length = len(data[0])

                pi = self.pi
                T = self.T
                mu = self.mu
                sigma = self.sigma
                M = self.M

                # TODO: won't work with empty sequences
                assert len(data) > 0
                assert np.array(data[0]).ndim == 2

                obj_array = np.zeros(len(data), dtype=np.object)
                for i, seq in enumerate(data):
                    # Matlab wants the sequences to have shape (n_dim, seq_length)
                    obj_array[i] = np.array(seq).T

                matlab_kwargs = dict(
                    data=obj_array, pi0=pi, T0=T, mu0=mu, sigma0=sigma, M0=M)

                try:
                    matlab_code = (
                        "run_mhmm('{{infile}}', '{{outfile}}', {max_iter}, "
                        "{thresh}, {verbose}, '{cov_type}'); exit;")
                    matlab_code = matlab_code.format(
                        max_iter=self.max_iter, thresh=self.thresh,
                        verbose=int(self.verbose), cov_type=self.cov_type)
                    matlab_results = run_matlab_code(
                        matlab_code, working_dir=self.directory,
                        verbose=self.verbose, delete_files=self.delete_matlab_files,
                        **matlab_kwargs)

                    log_likelihood = matlab_results['ll_trace'][0, -1]
                    if log_likelihood > 0:
                        raise Exception("Calculated positive likelihood.")
                    n_iters = matlab_results['ll_trace'].shape[1]

                    self.pi = matlab_results['pi'].reshape(-1)
                    self.T = matlab_results['T']
                    self.mu = matlab_results['mu']
                    self.sigma = matlab_results['sigma']
                    self.M = matlab_results['M']

                    self._validate_params()
                    break
                except Exception:
                    if n_attempts == self.max_attempts:
                        print(("%d failures, giving up." % n_attempts))
                        if self.raise_errors:
                            raise
                        else:
                            self._random_init(random_state)
                            matlab_results = dict(
                                pi=self.pi, T=self.T, mu=self.mu, sigma=self.sigma, M=self.M)
                            log_likelihood = -np.inf
                            break

            results.append((log_likelihood, matlab_results))

            print(("n_attempts: {0}".format(n_attempts)))
            print(("n_iters: {0}".format(n_iters)))
            print(("Final (total, not avg) log likelihood: {0}".format(log_likelihood)))

        best_results = max(results, key=lambda r: r[0])
        print(("Chose parameters with total log likelihood: %f" % best_results[0]))
        best_results = best_results[1]

        self.pi = best_results['pi'].reshape(-1)
        self.T = best_results['T']
        self.mu = best_results['mu']
        self.sigma = best_results['sigma']
        self.M = best_results['M']
        self._validate_params()

        self.reset()

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

        pi = self.pi
        T = self.T
        mu = self.mu
        sigma = self.sigma
        M = self.M

        # TODO: won't work with empty sequences
        assert len(data) > 0
        assert np.array(data[0]).ndim == 2

        obj_array = np.zeros(len(data), dtype=np.object)
        for i, seq in enumerate(data):
            # Matlab wants the sequences with shape (n_dim, seq_length)
            obj_array[i] = np.array(seq).T

        matlab_kwargs = dict(
            data=obj_array, pi=pi, T=T, mu=mu,
            sigma=sigma, M=M)

        matlab_code = "run_mhmm_cond_logprob('{infile}', '{outfile}');"
        results = run_matlab_code(
            matlab_code, working_dir=self.directory,
            delete_files=self.delete_matlab_files, **matlab_kwargs)

        # shape : (n_seqs, seq_length)
        cond_log_likelihood = results['cond_log_likelihood']
        return cond_log_likelihood

    def check_terminal(self, o):
        return False

    def _reset(self, initial=None):
        if self.careful:
            self._b = np.log(self.pi)
            b = np.exp(self._b)
            assert np.isclose(b.sum(), 1), "Should sum to 1: %s, %f" % (b, b.sum())
        else:
            self._b = self.pi
            assert np.isclose(self._b.sum(), 1), "Should sum to 1: %s, %f" % (self._b, self._b.sum())
        self._cond_obs_dist = None

    def update(self, o):
        old_b = self._b
        if self.careful:
            log_obs_prob = np.array([d.logpdf(o) for d in self.dists])
            log_obs_prob[np.isinf(log_obs_prob)] = -10000
            v = self._b.reshape(-1, 1) + self.log_T
            b = log_obs_prob + logsumexp(v, axis=0)
            self._b = b - logsumexp(b)
            b = np.exp(self._b)
            assert np.isclose(b.sum(), 1), (
                "Should sum to 1: %s, %f. "
                "Previous value: %s." % (b, b.sum(), np.exp(old_b)))
        else:
            b = self._b.dot(self.T) * np.array([d.pdf(o) for d in self.dists])
            self._b = normalize(b, ord=1)
            assert np.isclose(self._b.sum(), 1), (
                "Should sum to 1: %s, %f. "
                "Previous value: %s." % (self._b, self._b.sum(), old_b))
        self._cond_obs_dist = None

    def cond_obs_prob(self, o):
        """ Get probability of observing o given the current state. """
        # if self.careful:
        #     logprob = logsumexp(self._b + np.array([d.logpdf(o) for d in self.dists]))
        #     return logprob if log else np.exp(logprob)
        # else:
        #     prob = self._b.dot(np.array([d.pdf(o) for d in self.dists]))
        #     return np.log(prob) if log else prob
        return self.cond_obs_dist().pdf(o)

    def cond_termination_prob(self, o):
        return 0.0

    def cond_obs_dist(self):
        if self._cond_obs_dist is None:
            if self.careful:
                self._cond_obs_dist = MixtureDist(np.exp(self._b), self.dists)
            else:
                self._cond_obs_dist = MixtureDist(self._b, self.dists)

        return self._cond_obs_dist

    def cond_predict(self):
        most_likely = np.argmax(self._b)
        return self.dists[most_likely].largest_mode()

    def predicts(self, seq):
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
            log_prob += np.log(self.cond_obs_prob(o))
            self.update(o)

        if log:
            return log_prob
        else:
            return np.exp(log_prob)

    def prefix_prob(self, string, log=False):
        return self.string_prob(string, log)

    def RMSE(self, test_data):
        n_predictions = sum(len(seq) for seq in test_data)
        if n_predictions == 0:
            return np.inf

        se = 0.0
        for seq in test_data:
            predictions = self.predicts(seq)
            se += ((np.array(predictions) - np.array(seq))**2).sum()

        return np.sqrt(se / n_predictions)


def qualitative(data, labels, model=None, n_repeats=1, dir_name=None, prefix=None):
    # Find an example of each digit, plot how the network does on each.
    unique_labels = list(set(labels))

    digits_to_plot = []
    for l in unique_labels:
        n_digits = 0
        for i in np.random.permutation(list(range(len(labels)))):
            if labels[i] == l:
                digits_to_plot.append((data[i], labels[i]))
                n_digits += 1
                if n_digits == n_repeats:
                    break

    i = 0
    for digit, label in digits_to_plot:
        plt.figure()
        title = "label=%d_round=%d" % (label, i)
        if prefix:
            title = prefix + '_' + title

        plt.subplot(2, 1, 1)
        pendigits.plot_digit(digit, difference=True)
        plt.title(title)
        plt.subplot(2, 1, 2)
        if model is not None:
            model.reset()
            prediction = [model.cond_predict()]
            for o in digit:
                model.update(o)
                prediction.append(model.cond_predict())
            plt.title("Prediction")
            pendigits.plot_digit(prediction, difference=True)
        if dir_name is None:
            plt.show()
        else:
            plt.savefig(os.path.join(dir_name, title+'.png'))
        plt.close()
        i += 1


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    use_digits = [0]#, 1, 2]
    difference = True
    simplify = True

    shutil.rmtree('hmm_images')
    dir_name = 'hmm_images'
    os.makedirs(dir_name)

    max_data_points = None

    data, labels = pendigits.get_data(difference=difference, use_digits=use_digits, simplify=5, sample_every=3)
    labels = [l for ll in labels[:10] for l in ll]
    data = [d for dd in data[:10] for d in dd]

    if max_data_points is None:
        max_data_points = len(data)
    perm = np.random.permutation(len(data))[:max_data_points]

    labels = [labels[i] for i in perm]
    data = [data[i] for i in perm]
    print(("Amount of data: ", len(data)))

    pct_test = 0.2
    n_train = int((1-pct_test) * len(data))

    test_data = data[n_train:]
    test_labels = labels[n_train:]

    data = data[:n_train]
    labels = labels[:n_train]

    n_components = 1
    n_dim = 2

    for n_states in [10, 20, 30, 40, 50]:
        gmm_hmm = GmmHmm(
            n_states=n_states, n_components=n_components, n_dim=n_dim,
            max_iter=1000, thresh=1e-4, verbose=0, cov_type='full',
            directory="temp", random_state=None, careful=False, left_to_right=True, n_restarts=5)

        gmm_hmm.fit(data)
        print("Using %d states and digits: %s." % (n_states, use_digits))
        qualitative(
            data, labels, gmm_hmm, n_repeats=3,
            prefix='train_n_states=%d' % n_states, dir_name=dir_name)
        qualitative(
            test_data, test_labels, gmm_hmm, n_repeats=3,
            prefix='test_n_states=%d' % n_states, dir_name=dir_name)

        print(gmm_hmm.mean_log_likelihood(test_data))
        print(gmm_hmm.RMSE(test_data))

    # eps = gmm_hmm.sample_episodes(10, horizon=int(np.mean([len(s) for s in data])))
    # for s in eps:
    #     if s:
    #         pendigits.plot_digit(s, difference)
    #         plt.show()
