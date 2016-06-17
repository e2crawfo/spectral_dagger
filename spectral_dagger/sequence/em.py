import numpy as np
import copy
import subprocess
import os

from spectral_dagger.sequence import StochasticAutomaton

LIKE_ITS = 20000
MAX_VALID_ITERS = 1000
DELTA_THRESH = 0.1
NOT_IMPROVE = 3
IMPROVE_EPS = 0.1
VERBOSE = True
TMP_FILE = ".emtemp.fsm"

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

    def __init__(self, n_components, n_observations):
        self.n_components = n_components
        self.n_observations = n_observations

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

        self.observations = range(self.n_observations)

        self.B_o = {}
        for symbol in range(self.n_observations):
            self.B_o[symbol] = np.empty((self.n_components, self.n_components))

        self.B = sum(self.B_o.values())

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

        self.compute_start_end_vectors(b_0, b_inf_string, 'string')

        self.reset()

    def fit(self, data, valid_data=None):
        if not em_available():
            raise OSError("treba not found on system.")

        obs_file = ".obstmp"
        if not isinstance(data, str):
            self._write_obs_file(data, obs_file)
        else:
            obs_file = data

        # We do an initial batch of training no matter what.
        # Afterwards, we do additional rounds of training, stopping only once
        # we have gone some specified number of rounds without getting
        # improvement in terms of either WER or KL-divergence on the validation
        # set. If no validation set is provided, the additional rounds are not
        # done. The number of iterations in each round of training is given
        # by ``like_its``,  ``n_valid_its`` is the maximum number of additional
        # rounds of training.
        like_its = LIKE_ITS
        n_valid_its = 0 if valid_data is None else MAX_VALID_ITERS

        command = (
            "treba --train=bw --initialize=%d --max-delta=%f "
            "--restarts=5,%d --max-iter=1 %s" % (
                self.n_components, DELTA_THRESH, like_its, obs_file))
        treba_output = subprocess.check_output(command.split())

        n_not_improve = 0
        best_kl, best_wer = np.inf, np.inf
        best_model = copy.deepcopy(self)

        for i in range(n_valid_its):
            with open(TMP_FILE, "w") as fp:
                fp.write(treba_output)

            command = (
                "treba --train=bw --file=%s --max-iter=%d %s" % (
                    TMP_FILE, like_its, obs_file))
            treba_output = subprocess.check_output(command.split())

            self._parse_treba_model(treba_output)

            kl = self.get_perplexity(valid_data)  # or pautomac_score(self)
            wer = self.get_WER(valid_data)

            if VERBOSE:
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
