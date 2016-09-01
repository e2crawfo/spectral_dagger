import numpy as np

from spectral_dagger.utils import normalize, sample_multinomial
from spectral_dagger import Environment, Space


class MixtureStochAuto(Environment):
    def __init__(self, coefficients, stoch_autos):
        assert (0 <= coefficients).all()
        assert (1 >= coefficients).all()
        assert coefficients.ndim == 1
        assert np.isclose(coefficients.sum(), 1)
        self.coefficients = coefficients

        assert len(coefficients) == len(stoch_autos)
        self.stoch_autos = stoch_autos

        # Create a separate set of SA's for sampling since problems
        # can arise if a single object is used for both sampling and
        # filtering simultaneously
        self._sample_stoch_autos = [
            stoch_auto.deepcopy() for stoch_auto in stoch_autos]

        self.n_observations = stoch_autos[0].n_observations
        self.observations = stoch_autos[0].observations

        for stoch_auto in self.stoch_autos:
            assert stoch_auto.n_observations == self.n_observations

        self.reset()

    def __str__(self):
        s = "<MixtureStochAuto. coefficients: %s, n_obs: %d" % (
            self.coefficients, self.n_observations)

        for i, stoch_auto in enumerate(self.stoch_autos):
            s += ", %d: %s" % (i, stoch_auto)
        s += ">"

        return s

    def __repr__(self):
        return str(self)

    @property
    def action_space(self):
        return None

    @property
    def observation_space(self):
        return Space(set(self.stoch_autos[0].observations), "ObsSpace")

    @property
    def can_terminate(self):
        return all(
            stoch_auto.can_terminate() for stoch_auto in self.stoch_autos)

    def in_terminal_state(self):
        return self.choice.terminal

    def has_terminal_states(self):
        return self.stoch_autos[0].can_terminate()

    def has_reward(self):
        return False

    def reset(self, initial=None):
        # For filtering
        for stoch_auto in self.stoch_autos:
            stoch_auto.reset()
        self.state_dist = self.coefficients.copy()

        # For generating
        for stoch_auto in self._sample_stoch_autos:
            stoch_auto.reset()
        choice_idx = sample_multinomial(self.coefficients, self.random_state)
        self.choice = self._sample_stoch_autos[choice_idx]

    def step(self):
        if self.choice.terminal:
            return None
        o = self.choice.step()
        self.update(o)
        return o

    def update(self, o):
        """ Update state upon seeing an observation. """
        weights = np.array([
            stoch_auto.get_obs_prob(o) for stoch_auto in self.stoch_autos])
        self.state_dist = normalize(weights * self.state_dist, ord=1)

        for stoch_auto in self.stoch_autos:
            stoch_auto.update(o)

    def get_obs_prob(self, o):
        """ Get probability of observation for next time step. """
        weights = np.array([
            stoch_auto.get_obs_prob(o) for stoch_auto in self.stoch_autos])

        return (self.state_dist * weights).sum()

    def get_termination_prob(self):
        """ Get probability of terminating.  """
        weights = np.array([
            stoch_auto.get_termination_prob()
            for stoch_auto in self.stoch_autos])

        return (self.state_dist * weights).sum()

    def get_obs_dist(self):
        """ Get distribution over observations for next time step. """
        dist = [self.get_obs_prob(o) for o in self.observations]
        dist.append(self.get_termination_prob())
        dist = normalize(dist, ord=1)

        return dist

    def get_string_prob(self, string, log=True):
        """ Get probability of string. """
        weights = np.array([
            stoch_auto.get_string_prob(string)
            for stoch_auto in self.stoch_autos])

        p = (self.coefficients * weights).sum()
        return np.log(p) if log else p

    def get_delayed_string_prob(self, string, log=True):
        """ Get probability of string. """
        weights = np.array([
            stoch_auto.get_delayed_string_prob(string)
            for stoch_auto in self.stoch_autos])

        p = (self.coefficients * weights).sum()
        return np.log(p) if log else p

    def get_prefix_prob(self, prefix, log=True):
        """ Get probability of prefix. """
        weights = np.array([
            stoch_auto.get_prefix_prob(prefix)
            for stoch_auto in self.stoch_autos])

        p = (self.coefficients * weights).sum()
        return np.log(p) if log else p

    def get_delayed_prefix_prob(self, prefix, t, log=True):
        """ Get probability of observing prefix at a delay of ``t``.

        get_delayed_prefix_prob(p, 0) is equivalent to get_prefix_prob(p).

        """
        weights = np.array([
            stoch_auto.get_prefix_prob(prefix)
            for stoch_auto in self.stoch_autos])

        p = (self.coefficients * weights).sum()
        return np.log(p) if log else p

    def get_substring_expectation(self, substring):
        """ Get expected number of occurrences of a substring. """
        weights = np.array([
            stoch_auto.get_substring_expectation(substring)
            for stoch_auto in self.stoch_autos])

        return (self.coefficients * weights).sum()

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
