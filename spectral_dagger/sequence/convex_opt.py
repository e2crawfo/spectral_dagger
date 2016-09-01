import numpy as np
import ctypes as C
import logging
import os

import spectral_dagger.sequence
from spectral_dagger.sequence import StochasticAutomaton
from spectral_dagger.sequence import estimate_hankels, top_k_basis
from spectral_dagger.sequence.stochastic_automaton import MAX_BASIS_SIZE


logger = logging.getLogger(__name__)

MAXDIM = 50
machine_eps = np.finfo(float).eps


class ConvexOptSA(StochasticAutomaton):
    def __init__(self, n_observations, estimator='prefix'):
        self.b_0 = None
        self.b_inf = None
        self.B_o = None

        self._observations = range(n_observations)
        self.estimator = estimator

    def fit(self, data, tau, max_k=500,
            basis=None, hp_string=None, hankels=None, probabilistic=True,
            rank_tol=0.0):
        """ Fit a SA to the given data using a convex optimization algorithm.

        The algorithm solves a convex optimization problem by minimizing the
        regression error with a trace-norm penalty on the coefficient matrix.

        Some notes:
        Borja's thesis contains the details on the ADMM algorithm. It includes
        some constraints to ensure that the resulting ADMM is a well formed
        PNFA. Note that the normalization vector used to initialize the
        optimization must always be the vector of string probabilities.
        I believe this is because it is used in order to make sure that the
        normalization constraint is met, and the normalization constraint
        only works with the string normalization vector.

        Parameters
        ----------
        data: list of list
            Each sublist is a list of observations constituting a trajectory.
        tau: float > 0
            Regularization parameter controlling the extent to which the
            trace-norm of the solution is penalized.
        max_k: int > 0
            Maximum number of iterations to perform during optimization.
        estimator: string
            'string', 'prefix', or 'substring'.
        basis: length-2 tuple
            Contains prefix and suffix dictionaries.
        hp_string: ndarray
            String probabilities of prefixes in the basis. Used as
            initialization for the opimtization.
        hankels: length-(3 or 4) tuple
            Contains either (hs, hankel, symbol_hankels) or
            (hp, hs, hankel, symbol_hankels) in which case the first
            argument is ignored. If provided, then estimating the hankels
            matrices is skipped. If provided, a basis must also be provided.
        probabilistic: bool
            Whether to require that the model be a probabilistic
            finite automaton.
        rank_tol: float >= 0
            Singular values below this value will be thrown away.

        """
        if hankels:
            if not basis:
                raise ValueError(
                    "If `hankels` provided, must also provide a basis.")
            if len(hankels) == 3:
                hs, hankel_matrix, symbol_hankels = hankels
            elif len(hankels) == 4:
                _, hs, hankel_matrix, symbol_hankels = hankels
            else:
                raise NotImplementedError()
        else:
            if not basis:
                logger.debug("Generating basis...")
                basis = top_k_basis(data, MAX_BASIS_SIZE, estimator)

            logger.debug("Estimating Hankels...")
            hankels = estimate_hankels(
                data, basis, self.observations, estimator)
            _, hs, hankel_matrix, symbol_hankels = hankels

        if not hp_string:
            dummy_basis = (basis[0], {(): 0})
            hp_string, _, _, _ = estimate_hankels(
                data, dummy_basis, self.observations, 'string')
        hp_string = hp_string.squeeze()
        if hp_string.sum() < machine_eps:
            raise Exception(
                "String probability vector is all 0. "
                "Make sure the sampling horizon and basis are appropriate.")

        self.basis = basis
        self.hp_string = hp_string
        self.hankel = hankel_matrix
        self.symbol_hankels = symbol_hankels

        # load C library
        package_dir = os.path.dirname(spectral_dagger.sequence.__file__)
        lib = C.CDLL(os.path.join(package_dir, '_admm.so'))

        # NOTE use of FORTRAN or 'F' order when passing things to C.
        # The Eigen library uses this order.
        n_prefix, n_suffix = hankel_matrix.shape
        hankel_sigma = np.zeros(
            (self.n_observations * n_prefix, n_suffix),
            dtype=np.float64, order='F')

        # Convert collection of H_symbol matrices into stacked H_sigma matrix
        for i, o in enumerate(self.symbol_hankels):
            hankel_sigma[
                n_prefix*i:n_prefix*(i+1), :] = self.symbol_hankels[o]

        fortran_hankel = np.zeros(hankel_matrix.shape, order='F')
        fortran_hankel[:] = hankel_matrix

        # Initialize large stacked B matrix
        # and set b_inf to the prefix estimate of h.
        B = np.zeros(
            (self.n_observations * n_prefix, n_prefix),
            dtype=np.float64, order='F')

        # Get used as initialization for the optimization
        b_inf = np.zeros(n_prefix, dtype=np.float64, order='F')
        b_inf[:] = hp_string

        if probabilistic:
            f_learn_pnfa = lib.admm_pnfa_learn
            f_learn_pnfa.argtypes = [
                np.ctypeslib.ndpointer(dtype=np.float64),
                np.ctypeslib.ndpointer(dtype=np.float64),
                np.ctypeslib.ndpointer(dtype=np.float64),
                np.ctypeslib.ndpointer(dtype=np.float64),
                C.c_int, C.c_int, C.c_double, C.c_int, C.c_int]

            f_learn_pnfa.restype = C.c_int

            # Call C ADMM routine
            f_learn_pnfa(
                fortran_hankel, hankel_sigma, b_inf, B,
                n_prefix, self.n_observations, tau, max_k, MAXDIM)
        else:
            f_learn_wa = lib.admm_wfa_learn
            f_learn_wa.argtypes = [
                np.ctypeslib.ndpointer(dtype=np.float64),
                np.ctypeslib.ndpointer(dtype=np.float64),
                np.ctypeslib.ndpointer(dtype=np.float64),
                C.c_int, C.c_int, C.c_double, C.c_int, C.c_int]
            f_learn_wa.restype = C.c_int
            f_learn_wa(
                fortran_hankel, hankel_sigma, B,
                n_prefix, self.n_observations, tau, max_k, MAXDIM)

        B[B < machine_eps] = 0.0

        U, S, VT = np.linalg.svd(B)

        if rank_tol == 0:
            self.B_o = {}
            for i, o in enumerate(self.symbol_hankels):
                self.B_o[o] = B[n_prefix*i:n_prefix*(i+1), :n_prefix]

            self.B = sum(self.B_o.values())

            b_0 = np.zeros(n_prefix)
            b_0[0] = 1.
        else:
            n_sv_keep = max(np.count_nonzero(S > rank_tol), 1)
            U = U[:, :n_sv_keep]
            S = S[:n_sv_keep]
            VT = VT[:n_sv_keep, :]

            self.B_o = {}
            for i, o in enumerate(self.symbol_hankels):
                self.B_o[o] = VT.dot(B[n_prefix*i:n_prefix*(i+1), :].dot(VT.T))

            self.B = sum(self.B_o.values())

            b_0 = np.linalg.pinv(hankel_matrix.dot(VT.T)).dot(hs.squeeze())
            b_inf = VT.dot(b_inf)

        # b_inf will have been modified by the call to the c code.
        # b_inf = np.ones(n_prefix)

        self.compute_start_end_vectors(b_0, b_inf, 'string')

        self.reset()
