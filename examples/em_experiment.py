""" Fit an HMM in different ways. """
import subprocess
import numpy as np
import seaborn

from spectral_dagger.utils.experiment import (
    run_experiment_and_plot, Estimator, ExperimentSpec)
from spectral_dagger.envs.hmm import Chain
# from spectral_dagger.datasets.pautomac import make_pautomac_like
from spectral_dagger.sequence import ExpMaxSA

n_states = n_symbols = 5
horizon = np.inf
hmm = Chain(n_states, 0.1, stop_prob=0.2*np.ones(n_states))

seaborn.set(style="white")
seaborn.set_context(rc={'lines.markeredgewidth': 0.1})

random_state = np.random.RandomState(4)

# learn_halt = True
# horizon = np.inf if learn_halt else 5
# halts = 0.7 if learn_halt else 0
# alpha = 0.01
# beta = 0.5
# n_topics = 3
# n_symbols = 10
# n_words_per_doc = 25
# n_states = 5
# n_symbols = 3
# hmm = make_pautomac_like(
#     kind='hmm', n_states=n_states, n_symbols=n_symbols,
#     symbol_density=0.5, transition_density=0.5, alpha=1.0,
#     halts=halts, random_state=random_state)

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


def data_generator(n_train, n_test, random_state=None):
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


class ExpMax(SequenceEstimator):
    default_em_kwargs = dict(n_restarts=1, pct_valid=0.1, max_iters=10)

    def __init__(
            self, n_states=1, em_kwargs=None, max_iters=10,
            directory=".", name="ExpMax"):

        self._init(locals())

    def fit(self, X, y=None):
        em_kwargs = (
            self.default_em_kwargs if self.em_kwargs is None
            else self.em_kwargs)
        em_kwargs.update(max_iters=self.max_iters)

        try:
            self.stoch_auto_ = ExpMaxSA(
                self.n_states, X.n_symbols, directory=self.directory,
                verbose=False, **em_kwargs)
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
            'Log Likelihood',
            'Negative One Norm'],
        x_var_name='max_iters',
        x_var_values=None,
        x_var_display='\# Iters',
        n_repeats=5,
        kwargs=dict())

    quick_specs = dict(
        vs_n_train_samples=base_plot._replace(
            x_var_values=[10, 20, 30, 40],
            n_repeats=5))
    specs = dict(
        vs_n_train_docs=base_plot._replace(
            x_var_values=np.arange(1, 47, 5)))

    estimators = [
        ExpMax(em_kwargs=dict(n_restarts=1), name="1"),
        ExpMax(em_kwargs=dict(n_restarts=10), name="10"),
        ExpMax(em_kwargs=dict(n_restarts=20), name="20"),
        ExpMax(em_kwargs=dict(n_restarts=30), name="30"),
        ExpMax(em_kwargs=dict(n_restarts=40), name="40"),
        ExpMax(em_kwargs=dict(n_restarts=50), name="50")]
    # estimators = [
    #     ExpMax(em_kwargs=dict(pct_valid=0.0), name="0.0"),
    #     ExpMax(em_kwargs=dict(pct_valid=0.1), name="0.1"),
    #     ExpMax(em_kwargs=dict(pct_valid=0.2), name="0.2")]
    data_kwargs = dict(n_train=200, n_test=100)

    run_experiment_and_plot(
        'estimator', estimators, data_generator, specs, quick_specs,
        data_kwargs, search_kwargs=dict(n_iter=10),
        directory='/data/em_experiment/', params=params,
        random_state=random_state)
