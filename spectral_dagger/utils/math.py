import numpy as np

from spectral_dagger import process_rng


def sample_multinomial(p_vals, rng=None):
    rng = process_rng(rng)
    sample = rng.multinomial(1, p_vals)
    return np.where(sample > 0)[0][0]


def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


def geometric_sequence(base=np.e, tau=1.0, start=1):
    """ Implements a geometric sequence.  """

    i = 0.0
    while True:
        value = base**(start+i/tau)

        yield value
        i += 1


def p_sequence(start=1.0, p=1.0):
    """ Implements a p-sequence, a generalization of the harmonic sequence. """

    i = start
    while True:
        value = 1.0 / (i**p)
        yield value
        i += 1


def ndarray_to_string(a):
    s = ""

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            s += a[i, j]
        s += '\n'
    return s


def normalize(M, ord=2, axis=1, in_place=False, conservative=False):
    """ Normalize an ndarray along an axis.

    Parameters
    ----------
    M: ndarray
        The array to normalize.
    ord: {non-zero int, inf, -inf, 'fro', 'nuc'} (optional)
        The order of norm to use.
    axis: int
        Axis to normalize along.
    in_place: bool
        Whether to perform the normalization in-place, or return a copy.
    conservative: bool
        If set to True, will over-estimate the norm of each row/column.

    """
    if not isinstance(M, np.ndarray) or 'float' not in M.dtype.name:
        M = np.array(M, dtype='d')

    if M.ndim == 1:
        norm = np.linalg.norm(M, ord)
    else:
        new_shape = list(M.shape)
        new_shape[axis] = 1
        new_shape = tuple(new_shape)

        norm = np.linalg.norm(M, ord, axis).reshape(new_shape)

    if conservative:
        norm += 0.000001

    if in_place:
        M[:] = M / norm
    else:
        M = M.copy()
        M /= norm

    M[np.isnan(M)] = 0.0
    return M


def laplace_smoothing(alpha, X, data):
    """
    Laplace (or additive) smoothing of categorical data. Here we are
    estimating the parameters of a categorical random variable from
    observations of it. Corresponds to using a symmetric dirichlet
    distribution as a prior for the categorial parameters, with alpha
    as its parameter, and taking the expectation of the posterior.

    See: https://en.wikipedia.org/wiki/Additive_smoothing

    Parameters
    ----------
    alpha: float
        Smoothing parameter. 0 corresponds to no smoothing, 1 corresponds
        "add-one" smoothing.

    x: array-like
        The values the R.V. can take on.

    data: array-like
        Series of N observations of the R.V. under study.
    """

    m = {x: i for i, x in enumerate(X)}
    counts = np.bincount([m[d] for d in data + X])
    counts -= 1

    return (counts + alpha) / (len(data) + alpha * len(X))
