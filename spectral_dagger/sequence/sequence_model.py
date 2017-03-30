import abc
import numpy as np
import six
from scipy.stats import multivariate_normal

from spectral_dagger.base import Environment
from spectral_dagger.utils import normalize as _normalize
from spectral_dagger.utils.dists import Multinomial


def sample_words(n_samples, dist, words, rng):
    sample = rng.multinomial(n_samples, dist)
    sampled_words = []
    sampled_indices = []
    for idx, (s, word) in enumerate(zip(sample, words)):
        for i in range(s):
            sampled_words.append(word)
            sampled_indices.append(idx)
    return sampled_words, sampled_indices


def mean_log_likelihood(model, test_data, string=True):
    """ Get mean log likelihood assigned to the test data by the model. """
    if len(test_data) == 0:
        return -np.inf

    llh = 0.0

    for seq in test_data:
        if string:
            seq_llh = model.string_prob(seq, log=True)
        else:
            seq_llh = model.prefix_prob(seq, log=True)

        llh += seq_llh

    return llh / len(test_data)


def perplexity(model, test_data):
    """ Get model perplexity of the model on the test data.  """
    return np.exp(-model.mean_log_likelihood(test_data))


def mean_one_norm_error(model, test_data):
    """ Get average one-norm error of the model on the test data.

    1-norm-error is the 1-norm between the distribution over observations
    predicted by the model, and the distribution that places all its
    probability mass at the observation that was actually observed.

    """
    if len(test_data) == 0:
        return 2.0

    error = 0.0
    n_predictions = 0.0

    for seq in test_data:
        model.reset()

        for o in seq:
            prob = model.cond_obs_prob(o)
            model.update(o)
            error += 2 * (1 - prob)
            n_predictions += 1

        prob = model.cond_termination_prob()
        error += 2 * (1 - prob)
        n_predictions += 1

    return error / n_predictions


def WER(model, test_data):
    """ Get word error rate of the model on the test data. """
    if len(test_data) == 0:
        return 1.0

    n_errors = 0.0
    n_predictions = 0.0

    for seq in test_data:
        model.reset()

        for o in seq:
            prediction = model.cond_predict()
            n_errors += int(prediction != o)
            n_predictions += 1

            model.update(o)

        prediction = model.cond_predict()
        n_errors += int(prediction != model.n_observations)
        n_predictions += 1

    return n_errors / n_predictions


def RMSE(self, test_data):
    """ Get root mean squared error of the model on the test data. """
    if len(test_data) == 0:
        return np.inf

    error = 0.0
    n_predictions = 0.0

    for seq in test_data:
        self.reset()

        for o in seq:
            prediction = self.cond_predict()
            e = prediction - o
            error += e.dot(e)
            n_predictions += 1

            self.update(o)

    return np.sqrt(error / n_predictions)


@six.add_metaclass(abc.ABCMeta)
class SequenceModel(Environment):

    def _lookahead(self):
        """ Sample next observation/halt. """
        # Note: cond_obs_dist should set random state appropriately.
        dist = self.cond_obs_dist()
        o = dist.rvs()

        if self.check_terminal(o):
            self.terminal = True
        self.next_obs = o

    def reset(self, initial=None):
        self._reset(initial)
        self.terminal = False
        self._lookahead()

    def step(self):
        if self.terminal:
            return None

        o = self.next_obs
        self.update(o)
        self._lookahead()
        return o

    def in_terminal_state(self):
        return self.terminal

    @abc.abstractmethod
    def check_terminal(self, obs):
        """ Return whether the given observation signals termination. """
        raise NotImplementedError()

    @abc.abstractmethod
    def _reset(self, initial=None):
        """ Reset internal history. """
        raise NotImplementedError()

    @abc.abstractmethod
    def update(self, o):
        """ Update the internal state upon seeing an observation. """
        raise NotImplementedError()

    @abc.abstractmethod
    def cond_obs_prob(self, o):
        """ Probability of seeing ``o`` on next time step given history. """
        raise NotImplementedError()

    @abc.abstractmethod
    def cond_termination_prob(self):
        """ Probability of terminating given history.  """
        raise NotImplementedError()

    @abc.abstractmethod
    def cond_predict(self):
        """ The most likely next observation given the history. """
        raise NotImplementedError()

    @abc.abstractmethod
    def cond_obs_dist(self):
        """ Get distribution over observations for next time step,
            including termination observations.

        """
        raise NotImplementedError()

    @abc.abstractmethod
    def string_prob(self, string, log=False, conditional=False):
        raise NotImplementedError()

    @abc.abstractmethod
    def prefix_prob(self, prefix, log=False, conditional=False):
        raise NotImplementedError()

    def mean_log_likelihood(self, test_data, string=True):
        return mean_log_likelihood(self, test_data, string)

    def perplexity(self, test_data):
        return perplexity(self, test_data)

    def mean_one_norm_error(self, test_data):
        return mean_one_norm_error(self, test_data)

    def WER(self, test_data):
        return WER(self, test_data)

    def RMSE(self, test_data):
        return RMSE(self, test_data)


@six.add_metaclass(abc.ABCMeta)
class StaticSequenceModel(SequenceModel):
    """ Just reproduces and predicts the most recent observation. """
    def __init__(self, n_dim=None, n_obs=None):
        if (n_dim is not None) == (n_obs is not None):
            raise Exception("Supply exactly one of ``n_obs``, ``n_dim``.")
        self.n_dim = n_dim
        self.n_obs = n_obs
        self.n_observations = n_obs
        self.reset()

    def check_terminal(self, obs):
        return False

    def has_reward(self):
        return False

    def has_terminal_states(self):
        return False

    def observation_space(self):
        return None

    def action_space(self):
        return None

    def _reset(self, initial=None):
        """ Reset internal history. """
        raise NotImplementedError()

    def reset(self):
        self.prev_obs = None
        self._cond_obs_dist = None

    def update(self, o):
        self.prev_obs = o
        self._cond_obs_dist = None

    def cond_obs_prob(self, o):
        return self.cond_obs_dist().pdf(o)

    def cond_termination_prob(self):
        return 0

    def cond_predict(self):
        if self.n_obs is not None:
            return self.prev_obs if self.prev_obs is not None else 0
        else:
            return self.prev_obs if self.prev_obs is not None else np.zeros(self.n_dim)

    def cond_obs_dist(self):
        """ Get distribution over observations for next time step,
            including termination observations.

        """
        if self._cond_obs_dist is None:
            if self.n_obs is not None:
                if self.prev_obs is not None:
                    r = np.zeros(self.n_obs)
                    r[self.prev_obs] = 1.0
                else:
                    r = _normalize(np.ones(self.n_obs), ord=1)
                self._cond_obs_dist = Multinomial(r)
            else:
                mean = self.prev_obs if self.prev_obs is not None else np.zeros(self.n_dim)
                self._cond_obs_dist = multivariate_normal(mean=mean)
        return self._cond_obs_dist

    def string_prob(self, string, log=False):
        if self.n_obs is not None:
            return int(all(s == string[0] for s in string[1:]))
        else:
            self.reset()
            prob = 1
            for s in string:
                prob *= self.cond_obs_prob(s)
                self.update(s)

            return prob

    def prefix_prob(self, prefix, log=False):
        return self.string_prob(prefix, log)
