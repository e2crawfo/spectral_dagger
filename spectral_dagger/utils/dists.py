import six
import abc
import numpy as np
from scipy.stats import multivariate_normal, rv_discrete, bernoulli
from scipy.misc import logsumexp
from sklearn.utils import check_random_state


@six.add_metaclass(abc.ABCMeta)
class Distribution(object):

    @abc.abstractmethod
    def pdf(self, o):
        raise NotImplementedError()

    def logpdf(self, o):
        return np.log(self.pdf(o))

    @abc.abstractmethod
    def rvs(self, size=None, random_state=None):
        raise NotImplementedError()


class Multinomial(Distribution):
    def __init__(self, p, random_state=None):
        self.p = np.array(p)
        assert self.p.ndim == 1
        assert (self.p >= 0).all()
        assert np.isclose(self.p.sum(), 1)
        self.dist = rv_discrete(values=(range(len(p)), p))
        self.random_state = random_state

    def pdf(self, o):
        return self.p[o]

    def rvs(self, size=None, random_state=None):
        random_state = (
            self.random_state if random_state is None else random_state)
        random_state = check_random_state(random_state)
        return self.dist.rvs(size=size, random_state=random_state)

    def __getitem__(self, key):
        return self.pdf(key)

    def __array__(self):
        return self.p.copy()


class MixtureDist(Distribution):
    def __init__(self, pi, dists, random_state=None):
        assert np.isclose(sum(pi), 1)
        self.pi = pi
        self.log_pi = np.log(pi)
        self.pi_rv = rv_discrete(values=(range(len(pi)), pi))
        self.dists = dists
        self.random_state = random_state

    def pdf(self, o):
        return self.pi.dot([d.pdf(o) for d in self.dists])

    def logpdf(self, o):
        return logsumexp(self.log_pi + np.array([d.logpdf(o) for d in self.dists]))

    def rvs(self, size=None, random_state=None):
        random_state = (
            self.random_state if random_state is None else random_state)
        random_state = check_random_state(random_state)

        components = self.pi_rv.rvs(size=size, random_state=random_state)

        try:
            components = int(components)
            return self.dists[components].rvs(random_state=random_state)
        except (ValueError, TypeError):
            rvs = []
            for c in np.nditer(components):
                rvs.append(self.dists[c].rvs(random_state=random_state))
            rvs = np.array(rvs, dtype='object')
            rvs = rvs.reshape(*size)
            return rvs


class GMM(MixtureDist):
    def __init__(self, pi, means, covs):
        dists = [multivariate_normal(mean=m, cov=c) for m, c in zip(means, covs)]
        self.means = means
        self.covs = covs
        super(GMM, self).__init__(pi, dists)

    def largest_mode(self):
        return self.means[np.argmax(self.pi)]


class TerminationDist(Distribution):
    """ A continuous distribution augmented with a probability of terminating. """
    def __init__(self, dist, term_prob, term_symbol=None, random_state=None):
        self.dist = dist
        self.term_prob = term_prob
        self.term_symbol = term_symbol
        self.random_state = random_state

    def check_terminal(self, o):
        if isinstance(o, np.ndarray):
            return False
        else:
            return o == self.term_symbol

    def pdf(self, o):
        if self.check_terminal(o):
            return self.term_prob
        else:
            return (1 - self.term_prob) * self.dist.pdf(o)

    def rvs(self, size=None, random_state=None):
        random_state = (
            self.random_state if random_state is None else random_state)
        random_state = check_random_state(random_state)

        terminate = bernoulli(self.term_prob).rvs(size=size, random_state=random_state)

        try:
            int(terminate)
            if terminate:
                return self.term_symbol
            else:
                return self.dist.rvs(random_state=random_state)
        except:
            rvs = []
            for t in np.nditer(terminate):
                if t:
                    rvs.append(self.term_symbol)
                else:
                    v = self.dist.rvs(random_state=random_state)
                    rvs.append(v)
            rvs = np.array(rvs, dtype='object')
            rvs = rvs.reshape(*size)
            return rvs
