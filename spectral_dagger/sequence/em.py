import numpy as np
import copy
import subprocess
import os

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split

from spectral_dagger import Estimator
from spectral_dagger.sequence import StochasticAutomaton
from spectral_dagger.utils import remove_file

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


class ExpMaxSA(StochasticAutomaton, Estimator):
    """ Train a stochastic automaton using expectation maximization.

    We do an initial batch of training no matter what.
    Afterwards, if we are provided with validation data or if pct_valid > 0,
    then we do additional rounds of training, stopping only once
    we have gone some specified number of rounds without getting
    improvement in terms of either WER or KL-divergence on the validation
    data.

    Parameters
    ----------
    n_states: int > 0
        Number of dimensions of learned SA.
    n_observations: int > 0
        Number of operators.
    n_restarts: int > 0
        Number of restarts in the initial (non-validation) training step.
    max_iters: int > 0
        Maximum number of iterations for each restart.
    max_valid_rounds: int > 0
        Maximum number of validation rounds of training.
    max_delta: float >= 0
        Stop optimizing a restart once the improvement in log likelihood
        is less than this value.
    pct_valid: 1 >= float >= 0
        If validation is not supplied to the call to ``fit``, then this
        percent of the training data will be used as validation data.
    hmm: bool
        Whether to learn an HMM (if False, will learn a general PFA).
    alg: str
        String specifying an algorithm to use for fitting. See the
        *treba* man page for details.
    treba_args: str
        Other arguments, properly formatted, for the call to *treba*.
    directory: str
        Name of directory to store temporary files used for communicating
        with the *treba* package.
    verbose: bool
        If True, will print diagnostic information to stdout.

    """
    em_temp = ".emtemp.fsm"
    n_states = 0

    def __init__(
            self, n_states=1, n_observations=1,
            n_restarts=5, max_iters=10, max_valid_rounds=10, max_delta=0.5,
            pct_valid=0.0, hmm=False, alg="bw", treba_args="",
            directory='.', verbose=False):

        self.n_states = n_states
        self._observations = range(n_observations)
        self.n_restarts = n_restarts
        self.max_iters = max_iters
        self.max_valid_rounds = max_valid_rounds
        self.max_delta = max_delta
        self.pct_valid = pct_valid

        self.hmm = "--hmm" if hmm else ""

        assert alg in 'merge,mdi,bw,dabw,gs,vb,vit,vitbw'.split(','), (
            "The string ``%s`` does not correspond "
            "to any of treba's algorithms.")
        self.alg = alg

        self.treba_args = treba_args

        self.directory = directory
        self.verbose = verbose

    @property
    def record_attrs(self):
        return super(ExpMaxSA, self).record_attrs or set(['n_states'])

    def point_distribution(self, context):
        pd = super(ExpMaxSA, self).point_distribution(context)
        if 'max_states' in context:
            pd.update(n_states=range(2, context['max_states']))
        return pd

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
            self.B_o[symbol] = np.zeros((self.n_states, self.n_states))

        b_0 = np.zeros(self.n_states)
        b_0[0] = 1

        b_inf_string = np.zeros(self.n_states)

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
        if valid_data is None and self.pct_valid > 0.0:
            data, valid_data = train_test_split(
                data, test_size=self.pct_valid, random_state=self.rng)

        if valid_data is not None and len(valid_data) == 0:
            valid_data = None

        if not em_available():
            raise OSError("treba not found on system.")

        obs_file = os.path.join(self.directory, ".obstmp")
        if not isinstance(data, str):
            self._write_obs_file(data, obs_file)
        else:
            obs_file = data

        # --max-iter applies only to the final round of training.
        command = (
            "treba --train={alg} {hmm} --initialize={n_states} "
            "--restarts={n_restarts},{max_iters} --max-delta={max_delta} "
            "--max-iter={max_iters} {other} {obs_file}").format(
                alg=self.alg, hmm=self.hmm, n_states=self.n_states,
                n_restarts=self.n_restarts,
                max_iters=self.max_iters, max_delta=self.max_delta,
                other=self.treba_args, obs_file=obs_file)

        if self.verbose:
            print "Running treba with command: "
            print command
            treba_output = subprocess.check_output(command.split())
        else:
            with open(os.devnull, 'w') as FNULL:
                treba_output = subprocess.check_output(
                    command.split(), stderr=FNULL)

        self._parse_treba_model(treba_output)

        max_valid_rounds = 0 if valid_data is None else self.max_valid_rounds
        n_not_improve = 0
        best_kl, best_wer = np.inf, np.inf
        best_model = copy.deepcopy(self)

        with remove_file(self.em_temp):
            for i in range(max_valid_rounds):
                if self.verbose:
                    print ("Starting validation iteration %d" + "=" * 10) % i

                model_file = os.path.join(self.directory, self.em_temp)
                with open(model_file, "w") as fp:
                    fp.write(treba_output)

                command = (
                    "treba --train={alg} --max-delta={max_delta} "
                    "--max-iter={max_iters} --file={model_file} "
                    "{other} {obs_file}").format(
                        alg=self.alg, max_delta=self.max_delta,
                        max_iters=self.max_iters, model_file=model_file,
                        other=self.treba_args, obs_file=obs_file)

                if self.verbose:
                    print "Running treba with command: "
                    print command
                    treba_output = subprocess.check_output(command.split())
                else:
                    with open(os.devnull, 'w') as FNULL:
                        treba_output = subprocess.check_output(
                            command.split(), stderr=FNULL)

                self._parse_treba_model(treba_output)

                kl = self.perplexity(valid_data)  # or pautomac_score(self)
                wer = self.WER(valid_data)

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

        self.reset()

        return self


if __name__ == "__main__":
    from spectral_dagger.datasets import make_pautomac_like
    from spectral_dagger.sequence import SpectralSA

    n_states = 10
    n_symbols = 5

    pfa = make_pautomac_like('hmm', n_states, n_symbols, 0.5, 0.5, halts=0.1)
    train = pfa.sample_episodes(1000)
    test = pfa.sample_episodes(100)

    em_pfa = ExpMaxSA(n_states, n_symbols, n_restarts=1, pct_valid=0.1, verbose=True)
    em_pfa.fit(train)

    vit_em_pfa = ExpMaxSA(n_states, n_symbols, alg="vitbw", n_restarts=1, pct_valid=0.1, verbose=True)
    vit_em_pfa.fit(train)

    spectral_pfas = SpectralSA(n_states, n_symbols, estimator='string')
    spectral_pfas.fit(train)

    spectral_pfap = SpectralSA(n_states, n_symbols, estimator='prefix')
    spectral_pfap.fit(train)

    spectral_pfass = SpectralSA(n_states, n_symbols, estimator='substring')
    spectral_pfass.fit(train)

    print "EM Log Likelihood: ", em_pfa.mean_log_likelihood(test, string=True)
    print "Viterbi EM Log Likelihood: ", vit_em_pfa.mean_log_likelihood(test, string=True)
    print "Spectral Log Likelihood String: ", spectral_pfas.mean_log_likelihood(test, string=True)
    print "Spectral Log Likelihood Prefix: ", spectral_pfap.mean_log_likelihood(test, string=True)
    print "Spectral Log Likelihood Substring: ", spectral_pfass.mean_log_likelihood(test, string=True)
    print "Ground Truth Log Likelihood: ", pfa.mean_log_likelihood(test, string=True)

    print "EM WER: ", em_pfa.WER(test)
    print "Viterbi EM WER: ", vit_em_pfa.WER(test)
    print "Spectral WER String: ", spectral_pfas.WER(test)
    print "Spectral WER Prefix: ", spectral_pfap.WER(test)
    print "Spectral WER Substring: ", spectral_pfass.WER(test)
    print "Ground Truth WER: ", pfa.WER(test)
