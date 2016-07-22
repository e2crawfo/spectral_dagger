import numpy as np

from spectral_dagger.utils import normalize
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
        for stoch_auto in self.stoch_autos:
            stoch_auto.reset()
        for stoch_auto in self._sample_stoch_autos:
            stoch_auto.reset()

        self.choice = self.run_rng.choice(self._sample_stoch_autos)
        self.state_dist = self.coefficients.copy()

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

    def get_obs_dist(self):
        """ Get distribution over observations for next time step. """
        return np.array([self.get_obs_prob(o) for o in self.observations])

    def get_string_prob(self, string):
        """ Get probability of string. """
        weights = np.array([
            stoch_auto.get_string_prob(string)
            for stoch_auto in self.stoch_autos])

        return (self.state_dist * weights).sum()

    def get_delayed_string_prob(self, string):
        """ Get probability of string. """
        weights = np.array([
            stoch_auto.get_delayed_string_prob(string)
            for stoch_auto in self.stoch_autos])

        return (self.state_dist * weights).sum()

    def get_prefix_prob(self, prefix, init_state=None):
        """ Get probability of prefix. """
        weights = np.array([
            stoch_auto.get_prefix_prob(prefix)
            for stoch_auto in self.stoch_autos])

        return (self.state_dist * weights).sum()

    def get_delayed_prefix_prob(self, prefix, t, init_state=None):
        """ Get probability of observing prefix at a delay of ``t``.

        get_delayed_prefix_prob(p, 0) is equivalent to get_prefix_prob(p).

        """
        weights = np.array([
            stoch_auto.get_prefix_prob(prefix)
            for stoch_auto in self.stoch_autos])

        return (self.state_dist * weights).sum()

    def get_substring_expectation(self, substring):
        """ Get expected number of occurrences of a substring. """
        weights = np.array([
            stoch_auto.get_substring_expectation(substring)
            for stoch_auto in self.stoch_autos])

        return (self.state_dist * weights).sum()

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
            if base == 2:
                seq_llh = np.log2(self.get_string_prob(seq))
            else:
                seq_llh = np.log(self.get_string_prob(seq))

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
