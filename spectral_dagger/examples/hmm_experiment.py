""" Fit an HMM in different ways. """
import subprocess
import numpy as np
import seaborn

from spectral_dagger.utils.experiment import (
    run_experiment_and_plot, Estimator, ExperimentSpec)
from spectral_dagger.envs.hmm import Chain
from spectral_dagger.sequence import SpectralSA, CompressedSA, ExpMaxSA

seaborn.set(style="white")
seaborn.set_context(rc={'lines.markeredgewidth': 0.1})

n_states = 5
hmm = Chain(n_states, 0.1, stop_prob=0.2*np.ones(n_states))

params = locals()


class SequenceDataset(object):
    def __init__(self, data, n_symbols):
        self.data = np.array(data)
        self.n_symbols = n_symbols
        self.shape = self.data.shape

    @property
    def X(self):
        return self

    @property
    def y(self):
        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return SequenceDataset(self.data[key], self.n_symbols)


def data_generator(n_train, n_test, horizon=10, random_state=None):
    samples = hmm.sample_episodes(n_train + n_test, horizon=horizon)
    train = samples[:n_train]
    test = samples[n_train:]
    true_model = SequenceEstimator()
    true_model.stoch_auto_ = hmm
    true_model.name = "Ground Truth"
    return (SequenceDataset(train, hmm.n_observations),
            SequenceDataset(test, hmm.n_observations),
            {'max_states': 2 * hmm.n_states,
             'true_model': true_model})


class SequenceEstimator(Estimator):
    record_attrs = ['n_states']

    def point_distribution(self, context):
        return dict(n_states=np.arange(2, context['max_states']))


class Spectral(SequenceEstimator):
    def __init__(self, n_states=1, name="Spectral"):
        self._init(locals())

    def fit(self, X, y=None):
        self.stoch_auto_ = SpectralSA(self.n_states, X.n_symbols)
        self.stoch_auto_.fit(X.data)

        return self


class Compressed(SequenceEstimator):
    def __init__(self, n_states=1, name="Compressed"):
        self._init(locals())

    def fit(self, X, y=None):
        self.stoch_auto_ = CompressedSA(self.n_states, X.n_symbols)
        self.stoch_auto_.fit(X.data)

        return self


class ExpMax(SequenceEstimator):
    default_em_kwargs = dict(n_restarts=1, pct_valid=0.1, max_iters=10)

    def __init__(
            self, n_states=1, em_kwargs=None,
            directory=".", name="ExpMax"):

        self._init(locals())

    def fit(self, X, y=None):
        em_kwargs = (
            self.default_em_kwargs if self.em_kwargs is None
            else self.em_kwargs)

        try:
            self.stoch_auto_ = ExpMaxSA(
                self.n_states, X.n_symbols, directory=self.directory,
                **em_kwargs)
            self.stoch_auto_.fit(X.data)
        except subprocess.CalledProcessError as e:
            print "CalledProcessError: "
            print e.output
            raise e

        return self


def word_correct_rate(estimator, X, y=None):
    return 1 - estimator.stoch_auto_.WER(X.data)


def mean_log_likelihood(estimator, X, y=None):
    return estimator.stoch_auto_.mean_log_likelihood(X.data)


def mean_one_norm_score(estimator, X, y=None):
    return -estimator.stoch_auto_.mean_one_norm_error(X.data)

np.set_printoptions(threshold=10000, suppress=True)

if __name__ == "__main__":
    random_state = np.random.RandomState(4)

    base_plot = ExperimentSpec(
        title='Performance on Test Set',
        score=[word_correct_rate, mean_log_likelihood, mean_one_norm_score],
        score_display=[
            'Correct Prediction Rate',
            'Mean Log Likelihood',
            'Mean Negative One Norm'],
        x_var_name='n_train',
        x_var_values=None,
        x_var_display="\# of Training Samples",
        n_repeats=5,
        kwargs=dict())

    quick_specs = dict(
        vs_n_train_samples=base_plot._replace(
            x_var_values=[100, 600, 1100, 1600],
            n_repeats=1))
    specs = dict(
        vs_n_train_samples=base_plot._replace(
            x_var_values=[100, 600, 1100, 1600],
            n_repeats=5))

    estimators = [
        Spectral(), Compressed(),
        ExpMax(em_kwargs=dict(pct_valid=0.1), name='bw1'),
        ExpMax(em_kwargs=dict(pct_valid=0.1, alg='vit'), name='vit1'),
        ExpMax(em_kwargs=dict(pct_valid=0.1, alg='vitbw'), name='vitbw1'),
        ExpMax(em_kwargs=dict(pct_valid=0.1, alg='vb'), name='vb1'),
        ExpMax(em_kwargs=dict(pct_valid=0.0), name='bw'),
        ExpMax(em_kwargs=dict(pct_valid=0.0, alg='vit'), name='vit'),
        ExpMax(em_kwargs=dict(pct_valid=0.0, alg='vitbw'), name='vitbw'),
        ExpMax(em_kwargs=dict(pct_valid=0.0, alg='vb'), name='vb')]

    data_kwargs = dict(n_train=1000, n_test=1000, horizon=np.inf)
    search_kwargs = dict(n_iter=10, n_jobs=1)

    run_experiment_and_plot(
        'data', estimators, data_generator, specs, quick_specs,
        data_kwargs, search_kwargs=dict(n_iter=10),
        directory='/data/hmm_experiment/', params=params,
        random_state=random_state)
