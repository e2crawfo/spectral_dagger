import numpy as np

from spectral_dagger import Environment, Space
from spectral_dagger.utils import sample_multinomial
from spectral_dagger.spectral import PredictiveStateRep


class PAStringGenerator(Environment):
    """ An environment created from a probabilistic automaton. """

    def __init__(self, pa):
        self.observations = pa.observations
        self.can_terminate = pa.can_terminate
        self.terminal = False

        self.b_0 = pa.b_0.copy()

        self.B_o = {}
        for o in pa.B_o:
            self.B_o[o] = pa.B_o[o].copy()

        self.b_inf_string = pa.b_inf_string.copy()
        self.b_inf_prefix = pa.b_inf.copy()

        self.reset()

    @property
    def action_space(self):
        return None

    @property
    def observation_space(self):
        return Space(set(self.observations), "ObsSpace")

    @property
    def size(self):
        return self.b_0.size

    def in_terminal_state(self):
        return self.terminal

    def has_terminal_states(self):
        return self.can_terminate

    def has_reward(self):
        return False

    def lookahead(self):
        terminal_prob = self.b.dot(self.b_inf_string)
        self.terminal = self.rng.rand() < terminal_prob

        probs = np.array([
            self.b.dot(self.B_o[o]).dot(self.b_inf_prefix)
            for o in self.observations])

        # Normalize probs since we've already sampled whether to terminate.
        self.probs = probs / probs.sum()

    def reset(self, initial=None):
        self.b = self.b_0.copy()
        self.lookahead()

    def update(self):
        if self.terminal:
            return None

        sample = sample_multinomial(self.probs, self.rng)
        obs = self.observations[sample]

        numer = self.b.dot(self.B_o[obs])
        denom = numer.dot(self.b_inf_prefix)
        self.b = numer / denom

        self.lookahead()

        return obs

    def to_psr(self):
        psr = PredictiveStateRep(
            b_0=self.b_0, b_inf=self.b_inf_prefix, B_o=self.B_o,
            estimator='prefix')

        return psr


if __name__ == "__main__":
    from spectral_dagger.hmm.pautomac import parse_pautomac_file
    psr = parse_pautomac_file(1)
