import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
from sklearn.utils.extmath import randomized_svd
from copy import deepcopy
import logging

from spectral_dagger import LearningAlgorithm, Space, Policy
from spectral_dagger import get_model_rng
from spectral_dagger.sequence import hankel


logger = logging.getLogger(__name__)

machine_eps = np.finfo(float).eps
MAX_BASIS_SIZE = 100


class PredictiveStateRep(object):
    def __init__(self, b_0, b_inf, B_o, estimator):

        self.B_o = B_o
        self.B = sum(self.B_o.values())

        self.observations = B_o.keys()
        self.n_observations = len(self.observations)

        self.compute_start_end_vectors(b_0, b_inf, estimator)

        self.reset()

    def __str__(self):
        return ("<PredictiveStateRep. "
                "n_obs: %d, n_states: %d>" % (self.n_observations, self.size))

    def __repr__(self):
        return str(self)

    @property
    def action_space(self):
        return None

    @property
    def observation_space(self):
        return Space(set(self.observations), "ObsSpace")

    @property
    def can_terminate(self):
        return (self.b_inf_string != 0).any()

    @property
    def size(self):
        return self.b_0.size

    def reset(self):
        self.b = self.b_0.copy()

    def operator(self, o):
        return self.B_o[o]

    def update(self, o=None, a=None):
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

    def get_string_prob(self, string, init_state=None):
        """ Get probability of string. """

        if init_state is None:
            self.b = self.b_0.copy()
        else:
            self.b = init_state

        log_prob = 0.0

        for o in string:
            obs_prob = self.get_obs_prob(o)
            obs_prob = max(machine_eps, obs_prob)
            log_prob += np.log(obs_prob)
            self.update(o)

        end_prob = self.b.dot(self.b_inf_string)
        end_prob = max(machine_eps, end_prob)
        log_prob += np.log(end_prob)

        prob = np.e**log_prob
        return np.clip(prob, machine_eps, 1)

    def get_delayed_string_prob(self, string, t, init_state=None):
        """ Get probability of observing string at a delay of ``t``.

        get_delayed_string_prob(s, 0) is equivalent to get_string_prob(s).

        """
        if init_state is None:
            b = self.b_0.copy()
        else:
            b = init_state

        for i in range(t):
            b = b.dot(self.B)

        return self.get_string_prob(string, init_dist=b)

    def get_prefix_prob(self, prefix, init_state=None):
        """ Get probability of prefix. """

        if init_state is None:
            self.b = self.b_0.copy()
        else:
            self.b = init_state

        log_prob = 0.0

        for o in prefix:
            obs_prob = self.get_obs_prob(o)
            obs_prob = max(machine_eps, obs_prob)
            log_prob += np.log2(obs_prob)
            self.update(o)

        prob = 2**log_prob
        return np.clip(prob, machine_eps, 1)

    def get_delayed_prefix_prob(self, prefix, t, init_state=None):
        """ Get probability of observing prefix at a delay of ``t``.

        get_delayed_prefix_prob(p, 0) is equivalent to get_prefix_prob(p).

        """
        if init_state is None:
            b = self.b_0.copy()
        else:
            b = init_state

        for i in range(t):
            b = b.dot(self.B)

        return self.get_prefix_prob(prefix, init_dist=b)

    def get_substring_expectation(self, substring):
        """ Get expected number of occurrences of a substring. """

        self.b = self.b_0_substring.copy()

        log_prob = 0.0

        for o in substring:
            obs_prob = self.get_obs_prob(o)
            obs_prob = max(machine_eps, obs_prob)
            log_prob += np.log2(obs_prob)
            self.update(o)

        prob = 2**log_prob
        return np.clip(prob, machine_eps, 1)

    def get_WER(self, test_data):
        """ Get word error rate for the test data. """
        errors = 0.0
        n_predictions = 0.0

        for seq in test_data:
            self.reset()

            for o in seq:
                prediction = self.predict()
                self.update(o)

                if prediction != o:
                    errors += 1
                n_predictions += 1

        return errors / n_predictions

    def get_log_likelihood(self, test_data, base=2):
        """ Get average log likelihood for the test data. """
        llh = 0.0

        for seq in test_data:
            if self.can_terminate:
                if base == 2:
                    seq_llh = np.log2(self.get_string_prob(seq))
                else:
                    seq_llh = np.log(self.get_string_prob(seq))
            else:
                if base == 2:
                    seq_llh = np.log2(self.get_prefix_prob(seq))
                else:
                    seq_llh = np.log(self.get_prefix_prob(seq))

            llh += seq_llh

        return llh / len(test_data)

    def get_perplexity(self, test_data, base=2):
        """ Get model perplexity on the test data.  """

        return 2**(-self.get_log_likelihood(test_data, base=base))

    def get_1norm_error(self, test_data):
        error = 0.0
        n_predictions = 0.0

        for seq in test_data:
            self.reset()

            for o in seq:
                pd = np.array([
                    self.get_obs_prob(obs)
                    for obs in self.observations])
                pd = np.clip(pd, 0.0, np.inf)
                pd /= pd.sum()

                true_pd = np.zeros(len(self.observations))
                true_pd[o] = 1.0

                error += np.linalg.norm(pd - true_pd, ord=1)

                self.update(o)
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

            if (self.b_inf_string == 0).all():
                self.b_inf = np.ones_like(self.b_inf_string)
            else:
                self.b_inf = I_minus_B_inv.dot(self.b_inf_string)

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
        return PredictiveStateRep(
            self.b_0.copy(), self.b_inf.copy(),
            deepcopy(self.B_o), estimator='prefix')


class SpectralPSR(PredictiveStateRep):
    def __init__(self, observations):
        self.b_0 = None
        self.b_inf = None
        self.B_o = None

        self.observations = observations
        self.n_observations = len(self.observations)

    def fit(self, data, n_components, estimator='prefix',
            basis=None, svd=None, hankels=None, sparse=True):
        """ Fit a PSR to the given data using a spectral algorithm.

        Parameters
        ----------
        data: list of list of observations
            Each sublist is a list of observations constituting a trajectory.
        n_components: int
            Number of dimensions in feature space.
        estimator: string
            'string', 'prefix', or 'substring'.
        basis: length-2 tuple
            Contains prefix and suffix dictionaries.
        svd: length-3 tuple
            Contains U, Sigma, V^T, the SVD of a Hankel matrix. If provided,
            then computing the SVD of the estimated Hankel matrix is skipped.
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
                basis = hankel.top_k_basis(data, MAX_BASIS_SIZE, estimator)

            logger.debug("Estimating Hankels...")
            hankels = hankel.estimate_hankels(
                data, basis, self.observations, estimator, sparse=sparse)

            # Note: all hankels are scipy csr matrices
            hp, hs, hankel_matrix, symbol_hankels = hankels

        self.basis = basis

        self.hankel = hankel_matrix
        self.symbol_hankels = symbol_hankels

        n_components = min(n_components, hankel_matrix.shape[0])

        if svd:
            U, Sigma, VT = svd
        else:
            logger.debug("Performing SVD...")
            n_oversamples = 10
            n_iter = 5

            # H = U Sigma V^T
            U, Sigma, VT = randomized_svd(
                hankel_matrix, n_components, n_oversamples, n_iter)

        V = VT.T

        U = U[:, :n_components]
        V = V[:, :n_components]
        Sigma = np.diag(Sigma[:n_components])

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

        self.compute_start_end_vectors(b_0, b_inf, estimator)

        self.reset()


class CompressedPSR(PredictiveStateRep):
    def __init__(self, observations):
        self.b_0 = None
        self.b_inf = None
        self.B_o = None

        self.observations = observations
        self.n_observations = len(self.observations)

    def fit(
            self, data, n_components, noise_std=None, estimator='prefix',
            basis=None, phi=None, hankels=None):
        """ Fit a PSR to the given data using a compression algorithm.

        Parameters
        ----------
        data: list of list of observations
            Each sublist is a list of observations constituting a trajectory.
        n_components: int > 0
            Number of dimensions in feature space.
        noise_std: float > 0, optional
            Standard deviation of noise used to generate compression matrices.
            Defaults to 1 / sqrt(n_components).
        estimator: string
            'string', 'prefix', or 'substring'.
        basis: length-2 tuple
            Contains prefix and suffix dictionaries.
        phi: (n_suffix, n_components) ndarray, optional
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
                basis = hankel.top_k_basis(data, MAX_BASIS_SIZE, estimator)

            logger.debug("Estimating Hankels...")
            hankels = hankel.estimate_hankels(
                data, basis, self.observations, estimator, sparse=True)

            # Note: all hankels are scipy csr matrices
            hp, hs, hankel_matrix, symbol_hankels = hankels

        self.basis = basis
        self.n_components = n_components

        self.hankel = hankel_matrix
        self.symbol_hankels = symbol_hankels

        prefix_dict, suffix_dict = basis

        if phi is None:
            phi = get_model_rng().randn(len(suffix_dict), n_components)
            phi *= (1. / np.sqrt(n_components)
                    if noise_std is None else noise_std)

        self.phi = phi

        hp = np.zeros(len(prefix_dict))

        proj_hankel = np.zeros((len(prefix_dict), n_components))
        proj_sym_hankels = {}
        for obs in self.observations:
            proj_sym_hankels[obs] = np.zeros((len(prefix_dict), n_components))

        b_0 = np.zeros(n_components)

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

        self.compute_start_end_vectors(b_0, b_inf, estimator=estimator)

        self.reset()


class KernelInfo(object):
    """ Stores kernel information for a KernelPSR.

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
            ksum = k.sum()
            if ksum > 0:
                k = k / k.sum()

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


class KernelPSR(PredictiveStateRep):
    """
    Parameters
    ----------
    B_o: list of ndarray
        Contains operators for each of the kernel centers. It is a list
        wherein the order must be the same as the "kernel centers" list.

    """
    def __init__(
            self, b_0, b_inf, B_o, kernel_info, estimator):

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

    def B(self):
        if not hasattr(self, "_B") or self._B is None:
            self._B = (
                self.kernel_info.obs_kernel(np.zeros(self.obs_dim)) *
                sum(self.B_o))

        return self._B

    def update(self, o=None, a=None):
        """ Update state upon seeing an observation. """
        self._prediction_vec = None
        super(KernelPSR, self).update(o, a)

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


class SpectralKernelPSR(KernelPSR):
    def __init__(self, kernel_info):
        self.b_0 = None
        self.b_inf = None
        self.B_o = None

        self.kernel_info = kernel_info

        self.estimator = "prefix"

    def fit(self, data, n_components):
        """ Fit a KernelPSR to the given data using a spectral algorithm.

        Parameters
        ----------
        data: list of list
            Each sublist is a list of observations constituting a trajectory.
        n_components: int
            Number of dimensions in feature space.

        """
        logger.debug("Estimating Hankels...")
        hankels = hankel.estimate_kernel_hankels(
            data, self.kernel_info, self.estimator)

        hp, hankel_matrix, symbol_hankels = hankels

        self.hankel = hankel_matrix
        self.symbol_hankels = symbol_hankels

        n_components = min(n_components, hankel_matrix.shape[0])

        logger.debug("Performing SVD...")
        n_oversamples = 10
        n_iter = 5

        # H = U Sigma V^T
        U, Sigma, VT = randomized_svd(
            hankel_matrix, n_components, n_oversamples, n_iter)

        V = VT.T

        U = U[:, :n_components]
        V = V[:, :n_components]
        Sigma = np.diag(Sigma[:n_components])

        # P^+ = (HV)^+ = (USigma)^+ = Sigma^+ U+ = Sigma^-1 U.T
        P_plus = np.linalg.pinv(Sigma).dot(U.T)

        # S^+ = (V.T)^+ = V
        S_plus = V

        logger.debug("Computing operators...")
        self.B_o = [P_plus.dot(Ho).dot(S_plus) for Ho in symbol_hankels]
        self.B = (
            self.kernel_info.obs_kernel(np.zeros(self.obs_dim)) *
            sum(self.B_o))

        # b_0 S = hs => b_0 = hs S^+
        b_0 = hp.dot(V)

        # P b_inf = hp => b_inf = P^+ hp
        b_inf = P_plus.dot(hp)

        self.compute_start_end_vectors(b_0, b_inf, self.estimator)

        self._prediction_vec = None
        self.prediction_mat = np.array([
            bo.dot(self.b_inf) for bo in self.B_o]).T

        self.reset()


class SpectralPSRWithActions(object):
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
        basis = hankel.top_k_basis(data, max_basis_size)

        logger.debug("Estimating hankels...")

        # Note: all matrices returned by construct_hankels are csr_matrices
        if use_naive:
            logger.debug("...using naive estimator...")
            hankels = hankel.construct_hankels_with_actions(
                data, basis, self.actions, self.observations)

            hp, hs, hankel_matrix, symbol_hankels = hankels
        else:
            logger.debug("...using robust estimator...")
            hankels = hankel.construct_hankels_with_actions_robust(
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
            hankel_matrix, self.max_dim, n_oversamples, n_iter)

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

    def get_WER(self, test_data):
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

                self.update(o, a)

            llh += seq_llh

        return llh / len(test_data)


class SpectralClassifier(LearningAlgorithm):
    """
    A learning algorithm which learns to select actions in a
    POMDP setting based on observed trajectories and expert actions.
    Uses the observed trajectories to learn a PSR for the POMDP that
    gave rise to the trajectories, and then uses a classifier to learn
    a mapping between states of the PSR to expert actions.

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
        self.psr = SpectralPSRWithActions(pomdp.actions, pomdp.observations)

        self.psr.fit(trajectories, self.max_basis_size, self.n_components)

        psr_states = []
        flat_actions = []

        for t, response_actions in zip(trajectories, actions):
            self.psr.reset()

            for (a, o), response_action in zip(t, response_actions):
                psr_states.append(self.psr.b)
                flat_actions.append(response_action)

                self.psr.update(o, a)

        # Most sklearn predictors operate on strings or numbers
        action_lookup = {str(a): a for a in set(flat_actions)}
        str_actions = [str(a) for a in flat_actions]

        self.predictor.fit(psr_states, str_actions)

        def f(psr_state):
            psr_state = psr_state.reshape(1, -1)
            action_string = self.predictor.predict(psr_state)[0]
            return action_lookup[action_string]

        return SpectralPolicy(self.psr, f)


class SpectralPolicy(Policy):
    def __init__(self, psr, f):
        self.psr = psr
        self.f = f

    @property
    def action_space(self):
        return self.psr.action_space

    @property
    def observation_space(self):
        return self.psr.observation_space

    def reset(self, init_dist=None):
        if init_dist is not None:
            raise Exception(
                "Cannot supply initialization distribution to PSR. "
                "Only works with the initialization distribution "
                "on which it was trained.")

        self.psr.reset()

    def update(self, observation, action, reward=None):
        self.psr.update(observation, action)

    def get_action(self):
        return self.f(self.psr.b)


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

    print "Sampling trajectories..."
    trajectories = pomdp.sample_episodes(
        n_trajectories, exploration_policy, horizon)

    for use_naive in [True, False]:
        print "Training model..."

        psr = SpectralPSRWithActions(
            pomdp.actions, pomdp.observations, max_dim)

        b_0, B_ao, b_inf = psr.fit(
            trajectories, max_basis_size, n_components, use_naive)

        test_length = 10
        n_tests = 2000

        n_below = []
        top_three_count = 0

        print "Running tests..."

        display = False

        for t in range(n_tests):

            if display:
                print "\nStart test"
                print "*" * 20

            exploration_policy.reset()
            psr.reset()
            pomdp.reset()

            for i in range(test_length):
                action = exploration_policy.get_action()
                predicted_obs = psr.predict(action)

                pomdp_string = str(pomdp)

                actual_obs, _ = pomdp.update(action)

                rank = psr.get_obs_rank(action, actual_obs)

                n_below.append(rank[1])
                if rank[0] < 3:
                    top_three_count += 1

                psr.update(actual_obs, action)
                exploration_policy.update(actual_obs, action)

                if display:
                    print "\nStep %d" % i
                    print "*" * 20
                    print pomdp_string
                    print "Chosen action: ", action
                    print "Predicted observation: ", predicted_obs
                    print "Actual observation: ", actual_obs
                    print "PSR Rank of Actual Observation: ", rank

        print (
            "Average num below: ", np.mean(n_below),
            "of", len(pomdp.observations))
        print "Probability in top 3: %f" % (
            float(top_three_count) / (test_length * n_tests))

        n_test_trajectories = 40
        test_trajectories = []

        print "Sampling test trajectories for WER..."
        trajectories = pomdp.sample_episodes(
            n_test_trajectories, exploration_policy, horizon)

        print "Word error rate: ", psr.get_WER(test_trajectories)

        llh = psr.get_log_likelihood(test_trajectories, base=2)
        print "Average log likelihood: ", llh
        print "Perplexity: ", 2**(-llh)
