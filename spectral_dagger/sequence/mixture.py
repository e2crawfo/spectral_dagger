import numpy as np
from scipy.misc import logsumexp
from pprint import pformat

from spectral_dagger.utils import normalize, sample_multinomial
from spectral_dagger.sequence import SequenceModel
from spectral_dagger.utils.dists import MixtureDist


class MixtureSeqGen(SequenceModel):
    def __init__(self, coefficients, seq_gens):
        assert (0 <= coefficients).all(), pformat(coefficients)
        assert (1 >= coefficients).all(), pformat(coefficients)
        assert coefficients.ndim == 1, pformat(coefficients)
        assert np.isclose(coefficients.sum(), 1), pformat(coefficients)
        self.coefficients = coefficients

        assert len(coefficients) == len(seq_gens)
        self.seq_gens = seq_gens

        # Create a separate set of SG's for sampling and filtering
        # since problems can arise if a single object is used for
        # both sampling and filtering simultaneously
        # TODO: Make this work.
        # self._filter_seq_gens = [
        #     seq_gen.deepcopy() for seq_gen in seq_gens]
        # self._sample_seq_gens = [
        #     seq_gen.deepcopy() for seq_gen in seq_gens]
        self._filter_seq_gens = seq_gens
        self._sample_seq_gens = seq_gens

        self.reset()

    def __str__(self):
        s = "<MixtureSeqGen. coefficients: %s,\n" % self.coefficients

        for i, seq_gen in enumerate(self.seq_gens):
            s += ", %d: %s\n" % (i, seq_gen)
        s += ">"

        return s

    def __repr__(self):
        return str(self)

    @property
    def action_space(self):
        return None

    @property
    def observation_space(self):
        return self.sg.observation_space

    @property
    def n_observations(self):
        n_obs = getattr(self.seq_gens[0], 'n_observations', None)
        if n_obs is None:
            raise Exception(
                "Cannot determine ``n_observations`` for mixture model, "
                "component sequence generators are continuous.")
        return n_obs

    @property
    def can_terminate(self):
        return all(seq_gen.can_terminate for seq_gen in self.seq_gens)

    def in_terminal_state(self):
        return self.choice.terminal

    def has_terminal_states(self):
        return self.seq_gens[0].can_terminate

    def has_reward(self):
        return False

    def reset(self, initial=None):
        # For filtering
        for sg in self._filter_seq_gens:
            sg.reset()
        self.state_dist = self.coefficients.copy()

        # For generating
        for sg in self._sample_seq_gens:
            sg.reset()
        choice_idx = sample_multinomial(self.coefficients, self.random_state)
        self.choice = self._sample_seq_gens[choice_idx]

    def step(self):
        if self.choice.terminal:
            return None
        o = self.choice.step()
        self.update(o)
        return o

    def check_terminal(self, obs):
        """ Not implemented since we use our own method of generation. """
        raise NotImplementedError()

    def _reset(self, initial=None):
        """ Not implemented since we use our own method of generation. """
        raise NotImplementedError()

    # TODO: use log sum trick everywhere that it is necessary.
    def update(self, o):
        """ Update state upon seeing an observation. """
        weights = np.array([sg.cond_obs_prob(o) for sg in self._filter_seq_gens])
        self.state_dist = normalize(weights * self.state_dist, ord=1)

        for sg in self._filter_seq_gens:
            sg.update(o)

    def cond_obs_prob(self, o):
        """ Get probability of observation for next time step. """
        weights = np.array([sg.cond_obs_prob(o) for sg in self._filter_seq_gens])

        return (self.state_dist * weights).sum()

    def cond_termination_prob(self):
        """ Get probability of terminating.  """
        weights = np.array([
            sg.cond_termination_prob() for sg in self._filter_seq_gens])

        return (self.state_dist * weights).sum()

    def cond_predict(self):
        idx = np.argmax(self.state_dist)
        return self._filter_seq_gens[idx].cond_predict()

    def cond_obs_dist(self):
        """ Get distribution over observations for next time step,
            including termination observations.

        """
        return MixtureDist(self.state_dist,
                           [sg.cond_obs_dist() for sg in self._filter_seq_gens],
                           random_state=self.random_state)

    def string_prob(self, string, log=False):
        """ Get probability of string. """
        log_weights = np.array([
            sg.string_prob(string, log=True) for sg in self.seq_gens])
        log_prob = logsumexp(log_weights + np.log(self.coefficients))
        return log_prob if log else np.exp(log_prob)

    def delayed_string_prob(self, string, log=False):
        """ Get probability of string. """
        log_weights = np.array([
            sg.delayed_string_prob(string, log=True) for sg in self.seq_gens])
        log_prob = logsumexp(log_weights + np.log(self.coefficients))
        return log_prob if log else np.exp(log_prob)

    def prefix_prob(self, prefix, log=False):
        """ Get probability of prefix. """
        log_weights = np.array([
            sg.prefix_prob(prefix, log=True) for sg in self.seq_gens])
        log_prob = logsumexp(log_weights + np.log(self.coefficients))
        return log_prob if log else np.exp(log_prob)

    def delayed_prefix_prob(self, prefix, t, log=False):
        """ Get probability of observing prefix at a delay of ``t``.

        delayed_prefix_prob(p, 0) is equivalent to prefix_prob(p).

        """
        log_weights = np.array([
            sg.delayed_prefix_prob(prefix, log=True) for sg in self.seq_gens])
        log_prob = logsumexp(log_weights + np.log(self.coefficients))
        return log_prob if log else np.exp(log_prob)

    def substring_expectation(self, substring):
        """ Get expected number of occurrences of a substring. """
        weights = np.array([
            sg.substring_expectation(substring) for sg in self.seq_gens])

        return (self.coefficients * weights).sum()
