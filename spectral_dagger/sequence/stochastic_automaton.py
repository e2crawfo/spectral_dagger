from __future__ import print_function
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
from sklearn.utils.extmath import randomized_svd
from copy import deepcopy
import logging

from spectral_dagger import Environment, LearningAlgorithm, Space, Policy
from spectral_dagger.sequence import (
    estimate_hankels, estimate_kernel_hankels, top_k_basis,
    construct_hankels_with_actions, construct_hankels_with_actions_robust)
from spectral_dagger.utils import sample_multinomial
from spectral_dagger.utils import normalize as _normalize


logger = logging.getLogger(__name__)

machine_eps = np.finfo(float).eps
MAX_BASIS_SIZE = 100


class StochasticAutomaton(Environment):
    """ A Stochastic Automaton.

    Notes: Defines both ``update`` and ``step`` methods, so it can be either be
    used as an Environment or for filtering. Using a single instance for both
    purposes simultaneously requires some care, as the ``step`` method will
    alter the state used for filtering, and ``update`` will alter the state
    used for sampling.

    """
    def __init__(self, b_0, B_o, b_inf, estimator):

        self._observations = B_o.keys()

        self.B_o = B_o
        self.B = sum(self.B_o.values())

        self.compute_start_end_vectors(b_0, b_inf, estimator)
        self.reset()

    def __str__(self):
        return ("<StochasticAutomaton. "
                "n_obs: %d, n_states: %d>" % (self.n_observations,
                                              self.n_states))

    def __repr__(self):
        return str(self)

    @property
    def n_states(self):
        return self.b_0.size

    @property
    def n_observations(self):
        return len(self._observations)

    @property
    def observations(self):
        return self._observations

    @property
    def action_space(self):
        return None

    @property
    def observation_space(self):
        return Space(set(self.observations), "ObsSpace")

    @property
    def can_terminate(self):
        return (self.b_inf_string != 0).any()

    def in_terminal_state(self):
        return self.terminal

    def has_terminal_states(self):
        return self.can_terminate

    def has_reward(self):
        return False

    def operator(self, o):
        return self.B_o[o]

    def _lookahead(self):
        """ Decide whether to halt, and if not, compute next-state probs. """
        dist = self.get_obs_dist()
        terminal_prob = dist[-1]
        self.terminal = self.random_state.rand() < terminal_prob

        probs = dist[:-1]
        if probs.sum() == 0:
            raise Exception("Dividing by 0 when calculating next-step "
                            "probabilities for StochasticAutomaton.")

        # Normalize probs since we've already sampled whether to terminate.
        self.probs = probs / probs.sum()
        assert not any(np.isnan(self.probs))

    def reset(self, initial=None):
        self.b = self.b_0.copy()
        self._lookahead()

    def step(self):
        if self.terminal:
            return None

        sample = sample_multinomial(self.probs, self.random_state)
        o = self.observations[sample]
        self.update(o)
        self._lookahead()

        return o

    def update(self, o):
        """ Update state upon seeing an observation. """
        numer = self.b.dot(self.operator(o))
        denom = numer.dot(self.b_inf)

        if np.isclose(denom, 0):
            self.b = np.zeros_like(self.b)
        else:
            self.b = numer / denom

    def get_obs_prob(self, o):
        """ Get probability of observation for next time step.  """
        prob = self.b.dot(self.operator(o)).dot(self.b_inf)
        return np.clip(prob, machine_eps, 1)

    def get_termination_prob(self):
        """ Get probability of terminating.  """
        prob = self.b.dot(self.b_inf_string)
        return np.clip(prob, machine_eps, 1)

    def get_obs_dist(self):
        """ Get distribution over observations for next time step.

        The length of the returned array is ``n_observations + 1``.
        The final value in the array is the probability of halting.

        """
        dist = [self.get_obs_prob(o) for o in self.observations]
        dist.append(self.get_termination_prob())
        dist = _normalize(dist, ord=1)

        return dist

    def predict(self):
        """ Get observation with highest prob for next time step. """
        return max(self.observations, key=self.get_obs_prob)

    def get_obs_rank(self, o):
        """ Get probability rank of observation for next time step. """
        probs = np.array(
            [self.get_obs_prob(obs) for obs in self.observations])

        return (
            np.count_nonzero(probs > self.get_obs_prob(o)),
            np.count_nonzero(probs < self.get_obs_prob(o)))

    def get_string_prob(self, string, log=False, initial_state=None):
        """ Get probability of string. """

        if initial_state is None:
            self.b = self.b_0.copy()
        else:
            self.b = initial_state

        log_prob = 0.0

        for o in string:
            obs_prob = self.get_obs_prob(o)
            obs_prob = max(machine_eps, obs_prob)
            log_prob += np.log(obs_prob)
            self.update(o)

        end_prob = self.b.dot(self.b_inf_string)
        end_prob = max(machine_eps, end_prob)
        log_prob += np.log(end_prob)

        if log:
            return np.clip(log_prob, -np.inf, 0.0)
        else:
            prob = np.e**log_prob
            return np.clip(prob, machine_eps, 1)

    def get_delayed_string_prob(self, string, t, log=False):
        """ Get probability of observing string at a delay of ``t``.

        get_delayed_string_prob(s, 0) is equivalent to get_string_prob(s).

        """
        b = self.b_0.copy()
        for i in range(t):
            b = b.dot(self.B)

        return self.get_string_prob(string, log=log, initial_state=b)

    def get_prefix_prob(self, prefix, log=False, initial_state=None):
        """ Get probability of prefix. """
        if initial_state is None:
            self.b = self.b_0.copy()
        else:
            self.b = initial_state

        log_prob = 0.0

        for o in prefix:
            obs_prob = self.get_obs_prob(o)
            obs_prob = max(machine_eps, obs_prob)
            log_prob += np.log(obs_prob)
            self.update(o)

        if log:
            return np.clip(log_prob, -np.inf, 0.0)
        else:
            prob = np.e**log_prob
            return np.clip(prob, machine_eps, 1)

    def get_delayed_prefix_prob(self, prefix, t, log=False):
        """ Get probability of observing prefix at a delay of ``t``.

        get_delayed_prefix_prob(p, 0) is equivalent to get_prefix_prob(p).

        """
        b = self.b_0.copy()
        for i in range(t):
            b = b.dot(self.B)

        return self.get_prefix_prob(prefix, log=log, initial_state=b)

    def get_substring_expectation(self, substring):
        """ Get expected number of occurrences of a substring. """

        if not self.can_terminate():
            raise RuntimeError(
                "This stochastic automaton will never halt, so the "
                "expected number of occurrences of any substring will "
                "be infinite.")

        self.b = self.b_0_substring.copy()

        log_prob = 0.0

        for o in substring:
            obs_prob = self.get_obs_prob(o)
            obs_prob = max(machine_eps, obs_prob)
            log_prob += np.log2(obs_prob)
            self.update(o)

        prob = 2**log_prob
        return np.clip(prob, machine_eps, 1)

    def WER(self, test_data):
        """ Get word error rate for the test data. """
        n_errors = 0.0
        n_predictions = 0.0

        for seq in test_data:
            self.reset()

            for o in seq:
                dist = self.get_obs_dist()
                prediction = np.argmax(dist)

                n_errors += int(prediction != o)
                n_predictions += 1

                self.update(o)

            dist = self.get_obs_dist()
            prediction = np.argmax(dist)
            n_errors += int(prediction != self.n_observations)
            n_predictions += 1

        return n_errors / n_predictions

    def mean_log_likelihood(self, test_data, string=True):
        """ Get average log likelihood for the test data. """
        llh = 0.0

        for seq in test_data:
            if string:
                seq_llh = self.get_string_prob(seq, log=True)
            else:
                seq_llh = self.get_prefix_prob(seq, log=True)

            llh += seq_llh

        return llh / len(test_data)

    def perplexity(self, test_data):
        """ Get model perplexity on the test data.  """
        return np.exp(-self.mean_log_likelihood(test_data))

    def mean_one_norm_error(self, test_data):
        error = 0.0
        n_predictions = 0.0

        for seq in test_data:
            self.reset()

            for o in seq:
                dist = self.get_obs_dist()
                error += 2 * (1 - dist[o])
                self.update(o)
                n_predictions += 1

            dist = self.get_obs_dist()
            error += 2 * (1 - dist[self.n_observations])
            n_predictions += 1

        return error / n_predictions

    def compute_start_end_vectors(self, b_0, b_inf, estimator):
        """ Calculate other start and end vectors for all estimator types.

        Assumes self.B_o and self.B have already been set.

        Parameters
        ----------
        b_0: ndarray
            Start vector.
        b_inf: ndarray
            End vector.
        estimator: string
            The estimator that was used to calculate b_0 and b_inf.

        """
        b_0 = b_0.reshape(-1).copy()
        b_inf = b_inf.reshape(-1).copy()

        # See Lemma 6.1.1 in Borja Balle's thesis
        I_minus_B = np.eye(self.B.shape[0]) - self.B
        if estimator != 'substring':
            I_minus_B_inv = np.linalg.pinv(I_minus_B)

        if estimator == 'string':
            self.b_inf_string = b_inf

            # In most cases where an entry of b_inf is calculated
            # to be 0, it should actually be 1, corresponding to
            # summing over a distribution over a set with
            # uncountably many values.
            self.b_inf = I_minus_B_inv.dot(self.b_inf_string)
            self.b_inf[self.b_inf == 0] = 1.0

            self.b_0 = b_0
            self.b_0_substring = self.b_0.dot(I_minus_B_inv)

        elif estimator == 'prefix':
            self.b_inf = b_inf
            self.b_inf_string = I_minus_B.dot(self.b_inf)

            self.b_0 = b_0
            self.b_0_substring = self.b_0.dot(I_minus_B_inv)

        elif estimator == 'substring':
            self.b_inf = b_inf
            self.b_inf_string = I_minus_B.dot(self.b_inf)

            self.b_0_substring = b_0
            self.b_0 = self.b_0_substring.dot(I_minus_B)

        else:
            raise ValueError("Unknown Hankel estimator name: %s." % estimator)

    def deepcopy(self):
        return StochasticAutomaton(
            self.b_0.copy(), deepcopy(self.B_o),
            self.b_inf.copy(), estimator='prefix')


class SpectralSA(StochasticAutomaton):
    def __init__(self, n_states, n_observations, estimator='prefix'):
        self.b_0 = None
        self.b_inf = None
        self.B_o = None

        self._n_states = n_states
        self._observations = range(n_observations)
        self.estimator = estimator

    @property
    def n_states(self):
        return self.b_0.size if self.b_0 is not None else self._n_states

    def fit(self, data, basis=None, svd=None, hankels=None, sparse=False):
        """ Fit a SA to the given data using a spectral algorithm.

        Parameters
        ----------
        data: (list of sequences) or (sequence->probability) dictionary.
            In the first case, each sequence is a tuple or list of obs.
            In the second case, we have a dictionary where the keys are
            sequences stored as tuples of obs, which are mapped to the
            probability of the sequence occuring.
        basis: length-2 tuple
            Contains prefix and suffix dictionaries.
        svd: length-3 tuple
            Contains U, Sigma, V^T, the SVD of a Hankel matrix. If provided,
            then computing the SVD of the estimated Hankel matrix is skipped.
        hankels: length-4 tuple
            Contains hp, hs, hankel, symbol_hankels. If provided, then
            estimating the Hankel matrices is skipped. If provided, then
            a basis must also be provided.
        sparse: bool
            Whether to perform computation using sparse matrices.

        """
        if hankels:
            if not basis:
                raise ValueError(
                    "If `hankels` provided, must also provide a basis.")

            hp, hs, hankel_matrix, symbol_hankels = hankels
        else:
            if not basis:
                logger.debug("Generating basis...")
                basis = top_k_basis(data, MAX_BASIS_SIZE, self.estimator)

            logger.debug("Estimating Hankels...")
            hankels = estimate_hankels(
                data, basis, self.observations, self.estimator, sparse=sparse)

            # Note: all hankels are scipy csr matrices
            hp, hs, hankel_matrix, symbol_hankels = hankels

        self.basis = basis

        self.hankel = hankel_matrix
        self.symbol_hankels = symbol_hankels

        n_states = min(self.n_states, hankel_matrix.shape[0])

        if svd:
            U, Sigma, VT = svd
        else:
            logger.debug("Performing SVD...")
            n_oversamples = 10
            n_iter = 5

            # H = U Sigma V^T
            U, Sigma, VT = randomized_svd(
                hankel_matrix, n_states, n_oversamples, n_iter,
                random_state=self.random_state)

        V = VT.T

        U = U[:, :n_states]
        V = V[:, :n_states]
        Sigma = np.diag(Sigma[:n_states])

        # P^+ = (HV)^+ = (U Sigma)^+ = Sigma^+ U+ = Sigma^-1 U.T
        P_plus = np.linalg.pinv(Sigma).dot(U.T)
        if sparse:
            P_plus = csr_matrix(P_plus)

        # S^+ = (V.T)^+ = V
        S_plus = V
        if sparse:
            S_plus = csr_matrix(S_plus)

        logger.debug("Computing operators...")
        self.B_o = {}
        for o in self.observations:
            B_o = P_plus.dot(symbol_hankels[o]).dot(S_plus)
            if sparse:
                B_o = B_o.toarray()
            self.B_o[o] = B_o

        self.B = sum(self.B_o.values())

        # b_0 S = hs => b_0 = hs S^+
        b_0 = hs.dot(S_plus)
        if sparse:
            b_0 = b_0.toarray()[0, :]

        # P b_inf = hp => b_inf = P^+ hp
        b_inf = P_plus.dot(hp)
        if sparse:
            b_inf = b_inf.toarray()[:, 0]

        self.compute_start_end_vectors(b_0, b_inf, self.estimator)
        self.reset()

        return self


class CompressedSA(SpectralSA):
    def __init__(
            self, n_states, n_observations,
            noise_std=None, estimator='prefix'):

        self.noise_std = noise_std
        super(CompressedSA, self).__init__(n_states, n_observations, estimator)

    def fit(self, data, basis=None, phi=None, hankels=None):
        """ Fit a SA to the given data using a compression algorithm.

        Parameters
        ----------
        data: list of list of observations
            Each sublist is a list of observations constituting a trajectory.
        basis: length-2 tuple
            Contains prefix and suffix dictionaries.
        phi: (n_suffix, n_states) ndarray, optional
            A pre-computed projection matrix. If not provided, then a
            projection matrix is drawn randomly.
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
                basis = top_k_basis(data, MAX_BASIS_SIZE, self.estimator)

            logger.debug("Estimating Hankels...")
            hankels = estimate_hankels(
                data, basis, self.observations, self.estimator, sparse=True)

            # Note: all hankels are scipy csr matrices
            hp, hs, hankel_matrix, symbol_hankels = hankels

        self.basis = basis
        self.hankel = hankel_matrix
        self.symbol_hankels = symbol_hankels

        prefix_dict, suffix_dict = basis

        n_states = self.n_states

        if phi is None:
            phi = self.random_state.randn(len(suffix_dict), n_states)
            phi *= (1. / np.sqrt(n_states)
                    if self.noise_std is None else self.noise_std)

        self.phi = phi

        hp = np.zeros(len(prefix_dict))

        proj_hankel = np.zeros((len(prefix_dict), n_states))
        proj_sym_hankels = {}
        for obs in self.observations:
            proj_sym_hankels[obs] = np.zeros((len(prefix_dict), n_states))

        b_0 = np.zeros(n_states)

        for seq in data:
            for i in range(len(seq)+1):
                prefix = tuple(seq[:i])

                if prefix in suffix_dict:
                    b_0 += phi[suffix_dict[prefix], :]

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

        b_0 /= n_samples
        proj_hankel /= n_samples
        hp /= n_samples
        for o in self.observations:
            proj_sym_hankels[o] /= n_samples

        self.proj_sym_hankels = proj_sym_hankels
        self.proj_hankel = proj_hankel
        self.hp = hp

        inv_proj_hankel = np.linalg.pinv(proj_hankel)

        self.B_o = {}
        for o in self.observations:
            self.B_o[o] = inv_proj_hankel.dot(proj_sym_hankels[o])

        self.B = sum(self.B_o.values())

        b_inf = inv_proj_hankel.dot(hp)

        self.compute_start_end_vectors(b_0, b_inf, estimator=self.estimator)
        self.reset()

        return self


class KernelInfo(object):
    """ Stores kernel information for a KernelSA.

    Parameters
    ----------
    obs_kernel: function
        Kernel function for individual observations.
    obs_centers: (n_obs_centers, obs_dim) ndarray
        Kernel centers for individual observations.
    obs_kernel_grad: function
        Function which returns the gradient of the kernel.
    lmbda: positive float
        Bandwidth parameter. Only used for observations.

    prefix_kernel: function (optional)
        Kernel for prefixes. If not supplied, obs_kernel is used.
    prefix_centers: (n_prefix_centers, prefix_length * obs_dim) ndarray
        Kernel centers for prefixes. If not supplied, obs_centers is used.

    suffix_kernel: function (optional)
        Kernel for suffixes. If not supplied, obs_kernel is used.
    suffix_centers: (n_suffix_centers, suffix_length * obs_dim) ndarray
        Kernel centers for suffixes. If not supplied, obs_centers is used.

    """
    def __init__(self, obs_kernel, obs_centers, obs_kernel_grad, lmbda=1.0,
                 prefix_kernel=None, prefix_centers=None,
                 suffix_kernel=None, suffix_centers=None):

        self.obs_kernel = obs_kernel
        self.obs_centers = obs_centers
        self.obs_kernel_grad = obs_kernel_grad

        self.lmbda = lmbda

        self.prefix_kernel = (
            obs_kernel if prefix_kernel is None else prefix_kernel)
        self.prefix_centers = (
            obs_centers if prefix_centers is None else prefix_centers)

        self.suffix_kernel = (
            obs_kernel if suffix_kernel is None else suffix_kernel)
        self.suffix_centers = (
            obs_centers if suffix_centers is None else suffix_centers)

    @property
    def n_obs_centers(self):
        return len(self.obs_centers)

    @property
    def n_prefix_centers(self):
        return len(self.prefix_centers)

    @property
    def n_suffix_centers(self):
        return len(self.suffix_centers)

    @property
    def obs_dim(self):
        return self.obs_centers.shape[1] if self.obs_centers.ndim > 1 else 1

    @staticmethod
    def _eval_kernel(kernel, kernel_centers, o, normalize, lmbda):
        # Shape of kernel_centers: (n_centers, obs_dim)
        # Shape of o: (obs_dim,)
        obs_dim = kernel_centers.shape[1]

        if o.shape[0] != obs_dim:
            return None

        k = np.array([kernel((o - c) / lmbda) for c in kernel_centers])
        if normalize:
            k = _normalize(k, ord=1)

        return k

    def eval_obs_kernel(self, o, normalize=True):
        return self._eval_kernel(
            self.obs_kernel, self.obs_centers, o, normalize, self.lmbda)

    def eval_prefix_kernel(self, o, normalize=True):
        return self._eval_kernel(
            self.prefix_kernel, self.prefix_centers,
            o.flatten(), normalize, 1.0)

    def eval_suffix_kernel(self, o, normalize=True):
        return self._eval_kernel(
            self.suffix_kernel, self.suffix_centers,
            o.flatten(), normalize, 1.0)

    def obs_jacobian(self, o):
        jac = np.array([
            self.obs_kernel_grad((o - c) / self.lmbda)
            for c in self.obs_centers])
        return jac


class KernelSA(StochasticAutomaton):
    """
    Parameters
    ----------
    B_o: list of ndarray
        Contains operators for each of the kernel centers. It is a list
        wherein the order must be the same as the "kernel centers" list.

    """
    def __init__(self, b_0, B_o, b_inf, kernel_info, estimator):

        self.kernel_info = kernel_info

        self.b_0 = b_0
        self.b_inf = b_inf
        self.B_o = B_o

        self.estimator = estimator

        self.compute_start_end_vectors(b_0, b_inf, estimator)

        self._prediction_vec = None
        self.prediction_mat = np.array([
            bo.dot(self.b_inf) for bo in self.B_o]).T

        self.reset()

    @property
    def observation_space(self):
        return Space([-np.inf, np.inf] * self.obs_dim, "ObsSpace")

    def operator(self, o):
        sigma = self.kernel_info.eval_obs_kernel(o)
        B = sum([s * b for s, b in zip(sigma, self.B_o)])
        return B

    @property
    def obs_dim(self):
        return self.kernel_info.obs_dim

    @property
    def B(self):
        if not hasattr(self, "_B") or self._B is None:
            self._B = (
                self.kernel_info.obs_kernel(np.zeros(self.obs_dim)) *
                sum(self.B_o))

        return self._B

    def reset(self, initial=None):
        self.b = self.b_0.copy()

    def update(self, o=None, a=None):
        """ Update state upon seeing an observation. """
        self._prediction_vec = None
        super(KernelSA, self).update(o, a)

    @property
    def prediction_vec(self):
        if self._prediction_vec is None:
            self._prediction_vec = self.b.dot(self.prediction_mat)
        return self._prediction_vec

    def prediction_objective(self, o):
        """ Function to optimize to make predictions. """
        k = self.kernel_info.eval_obs_kernel(o)
        return self.prediction_vec.dot(k)

    def prediction_gradient(self, o):
        """ Gradient of the prediction function. """
        k = self.kernel_info.eval_obs_kernel(o, normalize=False)
        Z = k.sum()

        di_k = self.kernel_info.obs_jacobian(o).T
        di_Z = di_k.sum(axis=1) * (1/self.kernel_info.lmbda)

        n_centers = self.kernel_info.n_obs_centers

        term1 = di_k.dot(self.prediction_vec/(self.kernel_info.lmbda * Z))
        di_Z_diag_k = np.tile(di_Z, (n_centers, 1)).T * k
        term2 = di_Z_diag_k.dot(self.prediction_vec/(Z**2))
        return term1 - term2

    def predict(self):
        """ Get observation with highest prob for next time step. """
        def f(o):
            return -self.prediction_objective(o)

        # jac=False, so using numerical differentiation for now
        result = minimize(
            f, np.zeros(self.obs_dim), jac=False, method='Nelder-Mead')

        if not result.success:
            raise Exception(
                "Optimization for prediction failed. Output:\n %s" % result)

        return result.x

    def get_obs_rank(self, o):
        """ Get probability rank of observation for next time step. """
        raise NotImplementedError(
            "Cannot rank observations when observation space is cts.")


class SpectralKernelSA(KernelSA):
    def __init__(self, n_states, kernel_info):
        self.b_0 = None
        self.b_inf = None
        self.B_o = None

        self.kernel_info = kernel_info

        self._n_states = n_states
        self.estimator = "prefix"

    def n_states(self):
        return self.b_0.size if self.b_0 is not None else self._n_states

    def fit(self, data):
        """ Fit a KernelSA to the given data using a spectral algorithm.

        Parameters
        ----------
        data: list of list
            Each sublist is a list of observations constituting a trajectory.

        """
        logger.debug("Estimating Hankels...")
        hankels = estimate_kernel_hankels(
            data, self.kernel_info, self.estimator)

        hp, hankel_matrix, symbol_hankels = hankels

        self.hankel = hankel_matrix
        self.symbol_hankels = symbol_hankels

        n_states = min(self.n_states, hankel_matrix.shape[0])

        logger.debug("Performing SVD...")
        n_oversamples = 10
        n_iter = 5

        # H = U Sigma V^T
        U, Sigma, VT = randomized_svd(
            hankel_matrix, n_states, n_oversamples, n_iter,
            random_state=self.random_state)

        V = VT.T

        U = U[:, :n_states]
        V = V[:, :n_states]
        Sigma = np.diag(Sigma[:n_states])

        # P^+ = (HV)^+ = (USigma)^+ = Sigma^+ U+ = Sigma^-1 U.T
        P_plus = np.linalg.pinv(Sigma).dot(U.T)

        # S^+ = (V.T)^+ = V
        S_plus = V

        logger.debug("Computing operators...")
        self.B_o = [P_plus.dot(Ho).dot(S_plus) for Ho in symbol_hankels]

        # b_0 S = hs => b_0 = hs S^+
        b_0 = hp.dot(V)

        # P b_inf = hp => b_inf = P^+ hp
        b_inf = P_plus.dot(hp)

        self.compute_start_end_vectors(b_0, b_inf, self.estimator)

        self._prediction_vec = None
        self.prediction_mat = np.array([
            bo.dot(self.b_inf) for bo in self.B_o]).T

        self.reset()

        return self


# Below here is largely out of date.
class SpectralSAWithActions(object):
    def __init__(self, actions, observations, max_dim=80):

        self.n_actions = len(actions)
        self.actions = actions

        self.n_observations = len(observations)
        self.observations = observations

        self.B_ao = {}
        self.max_dim = max_dim

    @property
    def action_space(self):
        return Space(set(self.actions), "ActionSpace")

    @property
    def observation_space(self):
        return Space(set(self.observations), "ObsSpace")

    def fit(self, data, max_basis_size, n_components, use_naive=False):
        """
        data should be a list of lists. Each sublist corresponds to a
        trajectory.  Each entry of the trajectory should be a 2-tuple,
        giving the action followed by the observation.
        """

        logger.debug("Generating basis...")
        basis = top_k_basis(data, max_basis_size)

        logger.debug("Estimating hankels...")

        # Note: all matrices returned by construct_hankels are csr_matrices
        if use_naive:
            logger.debug("...using naive estimator...")
            hankels = construct_hankels_with_actions(
                data, basis, self.actions, self.observations)

            hp, hs, hankel_matrix, symbol_hankels = hankels
        else:
            logger.debug("...using robust estimator...")
            hankels = construct_hankels_with_actions_robust(
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
            hankel_matrix, self.max_dim, n_oversamples, n_iter,
            random_state=self.random_state)

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

    def update(self, obs, action):
        """Update state upon seeing an action observation pair"""
        B_ao = self.B_ao[action, obs]
        numer = self.b.dot(B_ao)
        denom = numer.dot(self.b_inf)

        if np.isclose(denom, 0):
            self.b = np.zeros_like(self.b)
        else:
            self.b = numer / denom

    def reset(self):
        self.b = self.b_0.copy()

    def predict(self, action):
        """
        Return the symbol that the model expects next,
        given that action is executed.
        """
        def predict(o):
            return self.get_obs_prob(action, o)
        return max(self.observations, key=predict)

    def get_obs_prob(self, a, o):
        """
        Returns the probablilty of observing o given
        we take action a in the current state. Interprets `o` as a
        prefix.
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

    def WER(self, test_data):
        """Returns word error rate for the test data"""
        errors = 0
        n_predictions = 0

        for seq in test_data:
            self.reset()

            for a, o in seq:
                prediction = self.predict(a)

                if prediction != o:
                    errors += 1

                self.update(o, a)

                n_predictions += 1

        return errors/float(n_predictions)

    def mean_log_likelihood(self, test_data, base=2):
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

                self.update(o, a)

            llh += seq_llh

        return llh / len(test_data)


class SpectralClassifier(LearningAlgorithm):
    """
    A learning algorithm which learns to select actions in a
    POMDP setting based on observed trajectories and expert actions.
    Uses the observed trajectories to learn a SA for the POMDP that
    gave rise to the trajectories, and then uses a classifier to learn
    a mapping between states of the SA to expert actions.

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
        self.sauto = SpectralSAWithActions(pomdp.actions, pomdp.observations)

        self.sauto.fit(trajectories, self.max_basis_size, self.n_components)

        sauto_states = []
        flat_actions = []

        for t, response_actions in zip(trajectories, actions):
            self.sauto.reset()

            for (a, o), response_action in zip(t, response_actions):
                sauto_states.append(self.sauto.b)
                flat_actions.append(response_action)

                self.sauto.update(o, a)

        # Most sklearn predictors operate on strings or numbers
        action_lookup = {str(a): a for a in set(flat_actions)}
        str_actions = [str(a) for a in flat_actions]

        self.predictor.fit(sauto_states, str_actions)

        def f(sauto_state):
            sauto_state = sauto_state.reshape(1, -1)
            action_string = self.predictor.predict(sauto_state)[0]
            return action_lookup[action_string]

        return SpectralPolicy(self.sauto, f)


class SpectralPolicy(Policy):
    def __init__(self, sauto, f):
        self.sauto = sauto
        self.f = f

    @property
    def action_space(self):
        return self.sauto.action_space

    @property
    def observation_space(self):
        return self.sauto.observation_space

    def reset(self, init_dist=None):
        if init_dist is not None:
            raise Exception(
                "Cannot supply initialization distribution to SA. "
                "An SA only works with the initialization distribution "
                "on which it was trained.")

        self.sauto.reset()

    def update(self, observation, action, reward=None):
        self.sauto.update(observation, action)

    def get_action(self):
        return self.f(self.sauto.b)


if __name__ == "__main__":
    from spectral_dagger.envs import EgoGridWorld
    from spectral_dagger.mdp import UniformRandomPolicy

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

    pomdp = EgoGridWorld(n_colors, world)

    exploration_policy = UniformRandomPolicy(pomdp.actions)
    trajectories = []

    print("Sampling trajectories...")
    trajectories = pomdp.sample_episodes(
        n_trajectories, exploration_policy, horizon)

    for use_naive in [True, False]:
        print("Training model...")

        sauto = SpectralSAWithActions(
            pomdp.actions, pomdp.observations, max_dim)

        b_0, B_ao, b_inf = sauto.fit(
            trajectories, max_basis_size, n_components, use_naive)

        test_length = 10
        n_tests = 2000

        n_below = []
        top_three_count = 0

        print("Running tests...")

        display = False

        for t in range(n_tests):

            if display:
                print("\nStart test")
                print("*" * 20)

            exploration_policy.reset()
            sauto.reset()
            pomdp.reset()

            for i in range(test_length):
                action = exploration_policy.get_action()
                predicted_obs = sauto.predict(action)

                pomdp_string = str(pomdp)

                actual_obs, _ = pomdp.update(action)

                rank = sauto.get_obs_rank(action, actual_obs)

                n_below.append(rank[1])
                if rank[0] < 3:
                    top_three_count += 1

                sauto.update(actual_obs, action)
                exploration_policy.update(actual_obs, action)

                if display:
                    print("\nStep %d", i)
                    print("*" * 20)
                    print(pomdp_string)
                    print("Chosen action: ", action)
                    print("Predicted observation: ", predicted_obs)
                    print("Actual observation: ", actual_obs)
                    print("SA Rank of Actual Observation: ", rank)

        print(
            "Average num below: ", np.mean(n_below),
            "of", len(pomdp.observations))
        print(
            "Probability in top 3: %f",
            float(top_three_count) / (test_length * n_tests))

        n_test_trajectories = 40
        test_trajectories = []

        print("Sampling test trajectories for WER...")
        trajectories = pomdp.sample_episodes(
            n_test_trajectories, exploration_policy, horizon)

        print("Word error rate: ", sauto.WER(test_trajectories))

        llh = sauto.mean_log_likelihood(test_trajectories, base=2)
        print("Average log likelihood: ", llh)
        print("Perplexity: ", 2**(-llh))
