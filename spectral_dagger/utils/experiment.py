from __future__ import print_function
import pprint
import logging
import abc
import six
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import uniform

import sklearn
from sklearn.cross_validation import train_test_split
from sklearn.metrics import make_scorer
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
from sklearn.utils import check_random_state

import spectral_dagger as sd
from spectral_dagger.utils.misc import make_symlink, make_filename

pp = pprint.PrettyPrinter()
verbosity = 2

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def update_attrs(obj, attrs):
    for k, v in six.iteritems(attrs):
        setattr(obj, k, v)


class Estimator(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def record_attrs(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def point_distribution(self, context, rng):
        raise NotImplementedError()


class Dataset(object):
    """ A generic dataset.

    Parameters/Attributes
    ---------------------
    X: any
        Input data.
    y: any
        Output data.

    """
    def __init__(self, X, y):
        if len(X) == 1:
            X = [X]
        self.X = X

        if len(y) == 1:
            y = [y]
        self.y = y

        assert len(self.X) == len(self.y)

    def __len__(self):
        return len(self.X)

    def __str__(self):
        return "<Dataset. len=%d>" % len(self)

    def __repr__(self):
        return str(self)

    def __iter__(self):
        return [(x, y) for x, y in zip(self.X, self.y)]

    def __getitem__(self, key):
        return Dataset(self.X[key], self.y[key])


class UnsupervisedDataset(Dataset):
    """ A dataset for unsupervised learning.

    Parameters/Attributes
    ---------------------
    X: any
        Data.

    """
    def __init__(self, X):
        if len(X) == 1:
            X = [X]
        self.X = X
        self.y = None

    def __iter__(self):
        return [x for x in self.X]

    def __getitem__(self, key):
        return UnsupervisedDataset(self.X[key])


class ExperimentDirectory(object):
    """ A directory for storing data from an experiment.

    Parameters
    ----------
    directory: string
        Name of directory in which to store the new experiment directory.
    exp_name: string
        Name of the experiment.
    data: dict
        Data to stringify and add to the ``exp_name``.
    use_time: bool
        Whether to add time information to ``exp_name`` (for
        identifying an experiment that ran at a given time).

    """
    def __init__(self, directory, exp_name, data=None, use_time=False):
        self.directory = directory
        self.exp_name = exp_name
        self.data = data
        self.use_time = use_time

        # TODO: include data, time
        self.exp_dir = make_filename(
            exp_name, use_time=use_time, config_dict=data)

        self.path = os.path.join(directory, self.exp_dir)
        os.makedirs(self.path)

        make_symlink(self.exp_dir, os.path.join(directory, 'latest'))

    def path_for_file(self, filename, subdir=""):
        """ Get a path for a file, creating necessary subdirs. """
        sd_path = os.path.join(self.path, subdir)
        if subdir and not os.path.isdir(sd_path):
            os.makedirs(sd_path)
        return os.path.join(self.path, subdir, filename)


def default_score(estimator, X, y):
    return estimator.score(X, y)


class Experiment(object):
    """ Run machine learning experiment.

        Two modes are available:
            data: Explores how the performance of one or more
                estimators changes as some parameter of the *data
                generation process* is varied.

            estimator: Explores how the performance of one or more
                estimators changes as some parameter of the *estimators*
                is varied (all tested estimators need to accept that
                parameter).

        Parameters
        ----------
        mode: one of ('data', 'estimator')
            Controls whether the x-variable is supplied to the data
            generation process or is used as an attribute on the estimators.
        base_estimators: Estimator class or list of Estimator classes.
            The base_estimators to be tested.
        x_var_name: string
            Name to use for the x variable (aka independent variable).
        x_var_values: list-like
            Values that the x variable can take on.
        generate_data: function
            A function which generates data. Must accept an argument with
            name equal to ``x_var_name``.
        score: None or function or list of function
            A function for evaluating a model. Signature: ``score(est, X, y)``.
            If None, then ``estimator.score`` is used. If a list of functions,
            then only the first is used for model selection; the others
            are evaluated on the selected models and recorded. Each entry
            may also be a tuple, in which case the first element of the tuple
            is a score function, and the second element is a name for the
            function. If names are not included, then the score function name
            is introspected.
        n_repeats: int
            Number of samples to take for each value of the x variable.
        data_kwargs: dict (optional)
            Key word arguments for the data generation function.
        search_kwargs: dict (optional)
            Key word arguments for the call to RandomizedSearchCV.
        exp_dir: str (optional)
            Name of an "experiments" directory. A new directory inside
            ``exp_dir`` will be created for this experiment, and relevant
            data will be stored in there (experimental results, plots, )
        training_stats: bool (optional)
            Whether to collect stats on the training performance of the best
            parameter setting on each search. Defaults to False.

    """
    def __init__(
            self, mode, base_estimators, x_var_name, x_var_values,
            generate_data, score=None, n_repeats=5, data_kwargs=None,
            search_kwargs=None, exp_dir='.', training_stats=False):

        assert mode in ['data', 'estimator']
        self.mode = mode
        self.base_estimators = base_estimators

        self.x_var_name = x_var_name
        self.x_var_values = x_var_values

        self.generate_data = generate_data

        if score is not None:
            if isinstance(score, tuple) or callable(score):
                score = [score]

            self.scores = []
            self.score_names = []

            for s in score:
                if isinstance(s, tuple):
                    self.scores.append(s[0])
                    self.score_names.append(s[1])
                else:
                    self.scores.append(s)
                    self.score_names.append(
                        s.__name__ if hasattr(s, "__name__") else str(s))
        else:
            self.scores = [default_score]
            self.score_names = ['score']
        self.score = self.scores[0]

        self.n_repeats = n_repeats
        self.data_kwargs = {} if data_kwargs is None else data_kwargs

        self.search_kwargs = dict(
            n_iter=10, n_jobs=4, cv=None, iid=False,
            error_score=np.inf, pre_dispatch='n_jobs')
        if search_kwargs is not None:
            self.search_kwargs.update(search_kwargs)

        self.exp_dir = exp_dir
        self.training_stats = training_stats

    def run(self, seed=None):
        """ Run the experiment.

        Parameters
        ----------
        seed: int or RandomState
            Random state for the experiment.

        """
        exp_dir = ExperimentDirectory(
            self.exp_dir, 'experiment', use_time=True)

        rng = check_random_state(seed)
        self.search_kwargs.update(dict(random_state=rng))

        results = []
        searches = defaultdict(list)
        for x in self.x_var_values:
            for i in range(self.n_repeats):
                data_kwargs = self.data_kwargs.copy()
                data_kwargs['seed'] = sd.gen_seed(rng)
                if self.mode == 'data':
                    data_kwargs[self.x_var_name] = x

                train, test = self.generate_data(**data_kwargs)

                for base_est in self.base_estimators:
                    est = sklearn.base.clone(base_est)
                    est.random_state = sd.gen_seed(rng)
                    if self.mode == 'estimator':
                        setattr(est, self.x_var_name, x)

                    print(
                        "Collecting data point. "
                        "method: %s, %s: %s, repeat: %d, seed: %d."
                        "" % (base_est.__class__.__name__, self.x_var_name,
                              str(x), i, est.random_state))

                    dists = est.point_distribution(
                        context=data_kwargs, rng=rng)

                    if self.mode == 'estimator':
                        dists.pop(self.x_var_name, None)

                    if dists:
                        search = RandomizedSearchCV(
                            est, dists, scoring=self.scores[0],
                            **self.search_kwargs)
                    else:
                        print ("``dists`` is an empty dictionary, "
                               "no hyper-parameters to select.")
                        search = GridSearchCV(
                            est, dists, scoring=self.scores[0],
                            cv=self.search_kwargs.get('cv', None))
                    search.fit(train.X, train.y)

                    # TODO: this won't work if multiple
                    # estimators have the same class
                    searches[base_est.__class__].append(search)
                    learned_est = search.best_estimator_

                    best_grid_score = max(
                        search.grid_scores_,
                        key=lambda gs: gs.mean_validation_score)

                    print("    Best parameter setting:"
                          " %s" % learned_est.get_params())

                    results.append({
                        'round': i, self.x_var_name: x,
                        'method': base_est.__class__.__name__})

                    for s, sn in zip(self.scores, self.score_names):
                        score = s(learned_est, test.X, test.y)
                        print("    Test score %s: %f" % (sn, score))
                        results[-1][sn] = score

                    if self.training_stats:
                        train_score = best_grid_score.mean_validation_score
                        train_score_std = np.std(
                            best_grid_score.cv_validation_scores)

                        print("    Training score: %f" % train_score)
                        results[-1]['training_score'] = train_score

                        print("    Training score std: %f" % train_score_std)
                        results[-1]['training_score_std'] = train_score_std

                    for attr in learned_est.record_attrs:
                        value = getattr(learned_est, attr)
                        try:
                            value = float(value)
                        except ValueError:
                            pass

                        results[-1][attr] = value
                        print("    Value for attr %s: %s" % (attr, value))

        self.searches = searches

        self.results = results
        self.df = pd.DataFrame.from_records(results)

        results_filename = exp_dir.path_for_file("results")
        self.df.to_csv(results_filename)

        return self.df


def data_experiment(display=False):
    """ Explore how lasso and ridge regression performance changes
    as the amount of training data changes. """

    import matplotlib.pyplot as plt
    from sklearn import datasets, linear_model
    from plot import plot_measures

    def generate_diabetes_data(train_size, seed):
        diabetes = datasets.load_diabetes()
        X_train, X_test, y_train, y_test = train_test_split(
            diabetes.data, diabetes.target,
            train_size=train_size, test_size=0.2, random_state=seed)

        return Dataset(X_train, y_train), Dataset(X_test, y_test)

    mse_score = make_scorer(
        sklearn.metrics.mean_squared_error, greater_is_better=False)
    mae_score = make_scorer(
        sklearn.metrics.mean_absolute_error, greater_is_better=False)

    class LogUniform(object):
        def __init__(self, low, high, base=np.e, random_state=None):
            assert low < high
            self.low = low
            self.high = high
            self.uniform = uniform(loc=low, scale=high-low)
            self.uniform.random_state = random_state
            self.base = base

        def rvs(self, size=1):
            return self.base ** self.uniform.rvs(size)

    class Ridge(linear_model.Ridge, Estimator):
        record_attrs = ['alpha']

        def point_grid(self, context, rng):
            return {'alpha': np.logspace(-4, -.5, 30)}

        def point_distribution(self, context, rng):
            return {'alpha': LogUniform(-4, -.5, 10, rng)}

    class Lasso(linear_model.Lasso, Estimator):
        record_attrs = ['alpha']

        def point_grid(self, context, rng):
            return {'alpha': np.logspace(-4, -.5, 30)}

        def point_distribution(self, context, rng):
            return {'alpha': LogUniform(-4, -.5, 10, rng)}

    # Insert into global namespace for pickling.
    globals()['Ridge'] = Ridge
    globals()['Lasso'] = Lasso

    experiment = Experiment(
        'data', [Ridge(0.1), Lasso(0.1)],
        'train_size', np.linspace(0.1, 0.8, 8),
        generate_diabetes_data, [(mse_score, 'NMSE'), (mae_score, 'NMAE')],
        search_kwargs=dict(cv=2, n_jobs=2), exp_dir='experiments')
    df = experiment.run()

    plt.figure(figsize=(10, 10))
    plot_measures(df, ['NMSE', 'NMAE', 'alpha'], 'train_size', 'method')
    if display:
        plt.show()

    return experiment, df


def estimator_experiment(display=False):
    """ Implement the sklearn LASSO example. """

    import matplotlib.pyplot as plt
    from sklearn import datasets, linear_model

    def generate_diabetes_data(seed=None):
        diabetes = datasets.load_diabetes()
        X_train = diabetes.data[:150]
        y_train = diabetes.target[:150]
        X_test = diabetes.data[150:]
        y_test = diabetes.target[150:]

        return Dataset(X_train, y_train), Dataset(X_test, y_test)

    class Ridge(linear_model.Ridge, Estimator):
        record_attrs = ['alpha']

        def point_distribution(self, context, rng):
            return {}

    class Lasso(linear_model.Lasso, Estimator):
        record_attrs = ['alpha']

        def point_distribution(self, context, rng):
            return {}

    # Insert into global namespace for pickling.
    globals()['Ridge'] = Ridge
    globals()['Lasso'] = Lasso

    alphas = np.logspace(-4, -.5, 30)
    experiment = Experiment(
        'estimator', [Ridge(), Lasso()], 'alpha', alphas,
        generate_diabetes_data, n_repeats=1,
        search_kwargs=dict(n_jobs=1), exp_dir='experiments',
        training_stats=True)

    df = experiment.run()
    scores = df[df['method'] == 'Lasso']['training_score'].values
    scores_std = df[df['method'] == 'Lasso']['training_score_std'].values

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
    e, df = data_experiment(True)
    e, df = estimator_experiment(True)
