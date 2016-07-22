""" Fit an HMM in different ways. """
import argparse
import os
import subprocess

import dill
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np
import seaborn
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle

from spectral_dagger.utils.experiment import Experiment, get_latest_exp_dir
from spectral_dagger.utils.plot import plot_measures
from spectral_dagger.envs.hmm import Chain
from spectral_dagger.sequence import SpectralSA, CompressedSA, ExpMaxSA

seaborn.set(style="white")
seaborn.set_context(rc={'lines.markeredgewidth': 0.1})

hmm = Chain(5, 0.1)


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


def generate_data(n_train, n_test, horizon=10, seed=None):
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


class SequenceEstimator(BaseEstimator):
    record_attrs = []

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def point_distribution(self, context, rng):
        return {}

    @property
    def directory(self):
        return self._directory

    @directory.setter
    def directory(self, directory):
        self._directory = directory
        if not os.path.isdir(directory):
            os.makedirs(directory)


class Spectral(SequenceEstimator):
    def __init__(self, n_components=1, name="Spectral"):
        self.n_components = n_components
        self.name = name

    def get_params(self, deep=True):
        return dict(n_components=self.n_components)

    def fit(self, X, y=None):
        self.stoch_auto_ = SpectralSA(self.n_components, X.n_symbols)
        self.stoch_auto_.fit(X.data)

        return self


class Compressed(SequenceEstimator):
    def __init__(self, n_components=1, name="Compressed"):
        self.n_components = n_components
        self.name = name

    def get_params(self, deep=True):
        return dict(n_components=self.n_components)

    def fit(self, X, y=None):
        self.stoch_auto_ = CompressedSA(self.n_components, X.n_symbols)
        self.stoch_auto_.fit(X.data)

        return self


class ExpMax(SequenceEstimator):
    def __init__(
            self, n_components=1, pct_for_validation=0.0,
            directory=".", name="ExpMax"):

        self.n_components = n_components
        self.pct_for_validation = pct_for_validation
        self.directory = directory
        self.name = name

    def get_params(self, deep=True):
        return dict(
            n_components=self.n_components,
            pct_for_validation=self.pct_for_validation,
            directory=self.directory)

    def fit(self, X, y=None):
        n_validate = int(len(X.data) * self.pct_for_validation)
        train = X.data[n_validate:]
        validate = X.data[:n_validate] if n_validate else None
        try:
            self.stoch_auto_ = ExpMaxSA(
                self.n_components, X.n_symbols, directory=self.directory,
                n_restarts=20)
            self.stoch_auto_.fit(train, validate)
        except subprocess.CalledProcessError as e:
            print "CalledProcessError: "
            print e.output
            raise e

        return self


def word_correct_rate(estimator, X, y=None):
    return 1 - estimator.stoch_auto_.get_WER(X.data)


def point_distribution(self, context, rng):
    assert 'max_states' in context, (
        "``context`` must specify ``max_states``.")
    n_components = np.arange(2, context['max_states'])
    d = {'n_components': n_components}
    return d


Spectral.point_distribution = point_distribution
Compressed.point_distribution = point_distribution
ExpMax.point_distribution = point_distribution


def run_experiment():
    directory = './hmm_exp'
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--plot', action='store_true',
        help='If True, will plot the most recent set of experiments '
             'rather running a new set.')
    parser.add_argument(
        '--show', action='store_true',
        help='If True, displays the plots as they are created.')
    args = parser.parse_args()

    mode = "data"
    title = "Sequence Prediction Performance"
    score = word_correct_rate
    score_display = "Word Correct Rate"
    x_var_name = "n_train"
    x_var_values = [100, 600, 1100, 1600]
    x_var_display = "\# of Training Samples"
    n_repeats = 5
    data_kwargs = dict(n_train=1000, n_test=1000, horizon=10)
    search_kwargs = dict(n_iter=10, n_jobs=1, error_score='raise')

    estimators = [Spectral(), Compressed()]#, ExpMax(pct_for_validation=0.1)]

    if not args.plot:
        experiment = Experiment(
                mode, estimators, x_var_name, x_var_values,
                generate_data, score,
                n_repeats=n_repeats, data_kwargs=data_kwargs,
                search_kwargs=search_kwargs, directory=directory,
                name=x_var_name)

        df = experiment.run('results_%s.csv' % x_var_name)
        exp_dir = experiment.exp_dir
        with open(exp_dir.path_for_file('experiment'), 'wb') as f:
            dill.dump(experiment, f, protocol=dill.HIGHEST_PROTOCOL)
    else:
        exp_dir = get_latest_exp_dir(directory)
        df = pd.read_csv(
            exp_dir.path_for_file('results_%s.csv' % x_var_name))
        with open(exp_dir.path_for_file('experiment'), 'rb') as f:
            experiment = dill.load(f)

    markers = MarkerStyle.filled_markers
    colors = seaborn.color_palette("hls", 16)

    labels = dict()

    def plot_kwarg_func(sv):
        idx = hash(str(sv))
        marker = markers[idx % len(markers)]
        c = colors[idx % len(colors)]
        label = labels.get(sv, sv)
        return dict(label=label, c=c, marker=marker)

    plt.figure(figsize=(5, 5))
    plot_measures(
        df, experiment.score_names, x_var_name, 'method',
        legend_outside=False,
        kwarg_func=plot_kwarg_func,
        measure_display=score_display,
        x_var_display=x_var_display)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=22)

    plt.gcf().subplots_adjust(left=0.1, right=0.95, top=0.94, bottom=0.12)
    plt.title(title)

    plt.gcf().set_size_inches(10, 5, forward=True)
    plt.savefig(exp_dir.path_for_file('plot_%s.pdf' % x_var_name))

    if args.show:
        plt.show()

if __name__ == "__main__":
    run_experiment()
