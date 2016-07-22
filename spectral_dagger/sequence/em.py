import numpy as np
import copy
import subprocess
import os

from spectral_dagger.sequence import StochasticAutomaton

MAX_VALID_ITERS = 1000
NOT_IMPROVE = 3
IMPROVE_EPS = 0.1

_treba_available = True

try:
    devnull = open(os.devnull, 'w')
    subprocess.call(["treba"], stdout=devnull, stderr=subprocess.STDOUT)
except OSError as e:
    if e.errno == os.errno.ENOENT:
        _treba_available = False
    else:
        raise


def em_available():
    return _treba_available


class ExpMaxSA(StochasticAutomaton):
    """ Train a stochastic automaton using expectation maximization.

    We do an initial batch of training no matter what.
    Afterwards, if we are provided with validation data then we do
    additional rounds of training, stopping only once
    we have gone some specified number of rounds without getting
    improvement in terms of either WER or KL-divergence on the validation
    set. The number of iterations in each round of training is given
    by ``n_iters``,  ``n_valid_iters`` is the maximum number of
    additional rounds of training.

    Parameters
    ----------
    n_components: int > 0
        Number of dimensions of learned SA.
    n_observations: int > 0
        Number of operators.
    n_restarts: int > 0
        Number of restarts in the initial (non-validation) training step.
    n_iters: int > 0
        Maximum number of iterations for each restart.
    delta_thresh: float > 0
        Stop optimizing a restart once the improvement in log likelihood
        is less than this value.
    directory: str
        Name of directory to store temporary files used for communicating
        with the treba package.
    verbose: bool
        If True, will print diagnostic information to stdout.

    """
    def __init__(
            self, n_components, n_observations,
            n_restarts=5, n_iters=100, delta_thresh=0.5,
            directory='.', verbose=False):

        self.n_components = n_components
        self._observations = range(n_observations)
        self.n_restarts = n_restarts
        self.n_iters = n_iters
        self.delta_thresh = delta_thresh
        self.directory = directory
        self.verbose = verbose

    @property
    def n_states(self):
        return self.n_components

    @staticmethod
    def _write_obs_file(samples, filename):
        with open(filename, 'w') as f:
            for t in samples:
                f.write(' '.join(str(int(o)) for o in t) + '\n')

    def _parse_treba_model(self, treba_output):
        """ Parse a string specifying a model in treba format.

        Populates the current instance with the parsed model.

        Parameters
        ----------
        treba_output: string
            Specification of a model in treba format.

        """
        treba_output_lines = treba_output.split("\n")

        self.B_o = {}
        for symbol in self.observations:
            self.B_o[symbol] = np.zeros((self.n_components, self.n_components))

        b_0 = np.zeros(self.n_components)
        b_0[0] = 1

        b_inf_string = np.zeros(self.n_components)

        for line in treba_output_lines:
            entries = line.split(' ')

            if len(entries) == 4:
                source_state = int(entries[0])
                target_state = int(entries[1])

                symbol = int(entries[2])
                prob = float(entries[3])
                self.B_o[symbol][source_state, target_state] = prob
            elif len(entries) == 2:
                source_state = int(entries[0])
                stop_prob = float(entries[1])
                b_inf_string[source_state] = stop_prob

        self.B = sum(self.B_o.values())

        self.compute_start_end_vectors(b_0, b_inf_string, 'string')

        self.reset()

    def fit(self, data, valid_data=None):
        if not em_available():
            raise OSError("treba not found on system.")

        obs_file = os.path.join(self.directory, ".obstmp")
        if not isinstance(data, str):
            self._write_obs_file(data, obs_file)
        else:
            obs_file = data
        n_valid_iters = 0 if valid_data is None else int(MAX_VALID_ITERS/self.n_components)
        n_iters = int(self.n_iters/self.n_components)

        command = (
            "treba --train=bw --initialize=%d --max-delta=%f "
            "--restarts=%d,%d --max-iter=1 %s" % (
                self.n_components, self.delta_thresh,
                self.n_restarts, n_iters, obs_file))

        if self.verbose:
            treba_output = subprocess.check_output(command.split())
        else:
            with open(os.devnull, 'w') as FNULL:
                treba_output = subprocess.check_output(
                    command.split(), stderr=FNULL)

        n_not_improve = 0
        best_kl, best_wer = np.inf, np.inf
        best_model = copy.deepcopy(self)

        for i in range(n_valid_iters):
            tmp_file = os.path.join(self.directory, ".emtemp.fsm")
            with open(tmp_file, "w") as fp:
                fp.write(treba_output)

            command = (
                "treba --train=bw --file=%s --max-iter=%d %s" % (
                    tmp_file, self.n_iters, obs_file))
            if self.verbose:
                treba_output = subprocess.check_output(command.split())
            else:
                with open(os.devnull, 'w') as FNULL:
                    treba_output = subprocess.check_output(
                        command.split(), stderr=FNULL)

            self._parse_treba_model(treba_output)

            kl = self.get_perplexity(valid_data)  # or pautomac_score(self)
            wer = self.get_WER(valid_data)

            if self.verbose:
                print "WER: ", wer, " KL:", kl

            if best_kl - kl < IMPROVE_EPS and best_wer - wer < IMPROVE_EPS:
                n_not_improve += 1
            else:
                n_not_improve = 0

                if kl < best_kl:
                    best_kl = kl

                if wer < best_wer:
                    best_wer = wer
                    best_model = copy.deepcopy(self)

            if n_not_improve >= NOT_IMPROVE:
                break

        self.best_model = best_model
        self.kl = best_kl
        self.wer = best_wer

        self._parse_treba_model(treba_output)
        self.reset()

        return self
