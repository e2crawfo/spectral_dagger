import numpy as np


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
    """ Implements a p sequence, a generalization of the harmonic sequence. """

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
