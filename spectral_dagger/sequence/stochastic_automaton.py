from __future__ import print_function
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
from sklearn.utils.extmath import randomized_svd
from copy import deepcopy
import logging

from spectral_dagger import Space
from spectral_dagger.sequence import (
    SequenceModel, Multinomial, estimate_hankels,
    estimate_kernel_hankels, top_k_basis)
from spectral_dagger.utils import normalize as _normalize


logger = logging.getLogger(__name__)

machine_eps = np.finfo(float).eps
MAX_BASIS_SIZE = 100


class StochasticAutomaton(SequenceModel):
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

    def has_terminal_states(self):
        return self.can_terminate

    def has_reward(self):
        return False

    def operator(self, o):
        return self.B_o[o]

    def check_terminal(self, obs):
        return obs == self.n_observations

    def _reset(self, initial=None):
        """ Reset internal history. """
        self.b = self.b_0.copy() if initial is None else initial
        self._cond_obs_dist = None

    def update(self, o):
        """ Update state upon seeing an observation. """
        numer = self.b.dot(self.operator(o))
        denom = numer.dot(self.b_inf)

        if np.isclose(denom, 0):
            self.b = np.zeros_like(self.b)
        else:
            self.b = numer / denom
        self._cond_obs_dist = None

    def cond_obs_dist(self):
        if self._cond_obs_dist is None:
            p = [self.b.dot(self.operator(obs)).dot(self.b_inf)
                 for obs in self.observations]

            termination_prob = self.b.dot(self.b_inf_string)
            p.append(termination_prob)

            p = [np.clip(_p, machine_eps, 1) for _p in p]

            p = _normalize(p, ord=1)
            self._cond_obs_dist = Multinomial(p)

        return self._cond_obs_dist

    def cond_obs_prob(self, o):
        """ Get probability of observation for next time step.  """
        return self.cond_obs_dist()[o]

    def cond_termination_prob(self):
        """ Get probability of terminating.  """
        return self.cond_obs_dist()[-1]

    def cond_predict(self):
        """ Get observation with highest prob for next time step. """
        return np.argmax(self.cond_obs_dist())

    def string_prob(self, string, log=False, initial_state=None):
        """ Get probability of string. """
        old_b = self.b

        self.reset(initial_state)

        log_prob = 0.0

        for o in string:
            obs_prob = self.cond_obs_prob(o)
            obs_prob = max(machine_eps, obs_prob)
            log_prob += np.log(obs_prob)
            self.update(o)

        end_prob = self.b.dot(self.b_inf_string)
        end_prob = max(machine_eps, end_prob)
        log_prob += np.log(end_prob)

        self.b = old_b

        if log:
            return np.clip(log_prob, -np.inf, 0.0)
        else:
            prob = np.e**log_prob
            return np.clip(prob, machine_eps, 1)

    def get_delayed_string_prob(self, string, t, log=False):
        """ Get probability of observing string at a delay of ``t``.

        get_delayed_string_prob(s, 0) is equivalent to string_prob(s).

        """
        b = self.b_0.copy()
        for i in range(t):
            b = b.dot(self.B)

        return self.string_prob(string, log=log, initial_state=b)

    def prefix_prob(self, prefix, log=False, initial_state=None):
        """ Get probability of prefix. """
        old_b = self.b

        self.reset(initial_state)

        log_prob = 0.0

        for o in prefix:
            obs_prob = self.cond_obs_prob(o)
            obs_prob = max(machine_eps, obs_prob)
            log_prob += np.log(obs_prob)
            self.update(o)

        self.b = old_b

        if log:
            return np.clip(log_prob, -np.inf, 0.0)
        else:
            prob = np.e**log_prob
            return np.clip(prob, machine_eps, 1)

    def get_delayed_prefix_prob(self, prefix, t, log=False):
        """ Get probability of observing prefix at a delay of ``t``.

        get_delayed_prefix_prob(p, 0) is equivalent to prefix_prob(p).

        """
        b = self.b_0.copy()
        for i in range(t):
            b = b.dot(self.B)

        return self.prefix_prob(prefix, log=log, initial_state=b)

    def get_substring_expectation(self, substring, initial_state=None):
        """ Get expected number of occurrences of a substring. """

        if not self.can_terminate:
            raise RuntimeError(
                "This stochastic automaton will never halt, so the "
                "expected number of occurrences of any substring will "
                "be infinite.")
        old_b = self.b
        if initial_state is None:
            self.b = self.b_0_substring.copy()
        else:
            self.b = initial_state

        log_val = 0.0

        for o in substring:
            obs_val = self.cond_obs_val(o)
            obs_val = max(machine_eps, obs_val)
            log_val += np.log(obs_val)
            self.update(o)

        self.b = old_b

        val = np.exp(log_val)
        return np.clip(val, machine_eps, 1)

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

    def _reset(self, initial=None):
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
