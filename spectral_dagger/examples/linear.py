
import pprint
import logging
import numpy as np
from scipy.stats import uniform
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model

import sklearn
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split
from sklearn.metrics import make_scorer

from spectral_dagger import Estimator
from spectral_dagger.utils.plot import plot_measures
from spectral_dagger.utils.experiment import Experiment, Dataset

pp = pprint.PrettyPrinter()
verbosity = 2

logger = logging.getLogger(__name__)


class LogUniform(object):
    def __init__(self, low, high, base=np.e, random_state=None):
        assert low < high
        self.low = low
        self.high = high
        self.uniform = uniform(loc=low, scale=high-low)
        self.uniform.random_state = random_state
        self.base = base

    def rvs(self, random_state=None, size=1):
        _rs = self.uniform.random_state
        if random_state is not None:
            self.uniform.random_state = random_state

        samples = self.base ** self.uniform.rvs(size)
        self.uniform.random_state = _rs
        return samples


class Ridge(linear_model.Ridge, Estimator):
    record_attrs = ['alpha']

    def __init__(self, alpha=1.0, random_state=None, name="Ridge"):
        super(Ridge, self).__init__(alpha=alpha, random_state=random_state)
        self.name = name

    def point_distribution(self, context):
        return {'alpha': LogUniform(-4, -.5, 10)}


class Lasso(linear_model.Lasso, Estimator):
    record_attrs = ['alpha']

    def __init__(self, alpha=1.0, random_state=None, name="Lasso"):
        super(Lasso, self).__init__(alpha=alpha, random_state=random_state)
        self.name = name

    def point_distribution(self, context):
        return {'alpha': LogUniform(-4, -.5, 10)}


def data_experiment(directory=None, display=False):
    """ Explore how lasso and ridge regression performance changes
    as the amount of training data changes. """

    def generate_diabetes_data(train_size, random_state):
        diabetes = datasets.load_diabetes()
        X_train, X_test, y_train, y_test = train_test_split(
            diabetes.data, diabetes.target,
            train_size=train_size, test_size=0.2, random_state=random_state)

        return Dataset(X_train, y_train), Dataset(X_test, y_test)

    mse_score = make_scorer(
        sklearn.metrics.mean_squared_error, greater_is_better=False)
    mae_score = make_scorer(
        sklearn.metrics.mean_absolute_error, greater_is_better=False)

    experiment = Experiment(
        'data', [Ridge(0.1), Lasso(0.1)],
        'train_size', np.linspace(0.1, 0.8, 8),
        generate_diabetes_data, [(mse_score, 'NMSE'), (mae_score, 'NMAE')],
        search_kwargs=dict(cv=2, n_jobs=2), directory=directory,
        use_time=False)
    df = experiment.run()

    plt.figure(figsize=(10, 10))
    plot_measures(df, ['NMSE', 'NMAE', 'alpha'], 'train_size', 'method')
    if display:
        plt.show()

    return experiment, df


class Ridge2(Ridge):
    def point_distribution(self, context):
        return {}


class Lasso2(Lasso):
    def point_distribution(self, context):
        return {}


def estimator_experiment(directory=None, display=False):
    """ Implement the sklearn LASSO example. """

    def generate_diabetes_data(random_state=None):
        diabetes = datasets.load_diabetes()
        X_train = diabetes.data[:150]
        y_train = diabetes.target[:150]
        X_test = diabetes.data[150:]
        y_test = diabetes.target[150:]

        return Dataset(X_train, y_train), Dataset(X_test, y_test)

    alphas = np.logspace(-4, -.5, 30)
    experiment = Experiment(
        'estimator', [Ridge(), Lasso()], 'alpha', alphas,
        generate_diabetes_data, n_repeats=1,
        search_kwargs=dict(n_jobs=1), directory=directory,
        save_datasets=True, save_estimators=True, use_time=False)

    df = experiment.run()
    scores = df[df['method'] == 'Lasso']['cv_score'].values
    scores_std = df[df['method'] == 'Lasso']['cv_score_std'].values

    plt.figure(figsize=(4, 3))
    plt.semilogx(alphas, scores)
    # plot error lines showing +/- std. errors of the scores
    plt.semilogx(
        alphas, np.array(scores) + np.array(scores_std) / np.sqrt(150), 'b--')
    plt.semilogx(
        alphas, np.array(scores) - np.array(scores_std) / np.sqrt(150), 'b--')
    plt.ylabel('CV score')
    plt.xlabel('alpha')

    if display:
        plt.show()

    return experiment, df


if __name__ == "__main__":
    e, df = data_experiment(display=True)
    e, df = estimator_experiment(display=True)
