from __future__ import print_function
import pprint
import logging
import abc
import six
import os
from collections import defaultdict
import time
import numpy as np
import pandas as pd
from collections import namedtuple
import argparse
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import seaborn
import dill
import sys
import dill

import sklearn
from sklearn.base import BaseEstimator
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
from sklearn.utils import check_random_state

import spectral_dagger as sd
from spectral_dagger.utils.misc import (
    make_symlink, make_filename, indent, as_title)
from spectral_dagger.utils.plot import plot_measures

pp = pprint.PrettyPrinter()
verbosity = 2

logger = logging.getLogger(__name__)


@six.add_metaclass(abc.ABCMeta)
class Estimator(BaseEstimator):
    """ If deriving classes define an attribute ``record_attrs``
        giving a list of strings, then those attributes will be queried
        by the Experiment class every time a cross-validation search
        finishes. Can be useful for recording the values of hyper-parameters
        chosen by cross-validation.

    """
    record_attrs = []

    def _init(self, _locals):
        if 'self' in _locals:
            del _locals['self']

        for k, v in six.iteritems(_locals):
            setattr(self, k, v)

    def get_estimated_params(self):
        estimated_params = {}
        for attr in dir(self):
            if attr.endswith('_') and not attr.startswith('_'):
                estimated_params[attr] = getattr(self, attr)
        return estimated_params

    @abc.abstractmethod
    def point_distribution(self, context):
        raise NotImplementedError()

    @property
    def directory(self):
        return self._directory

    @directory.setter
    def directory(self, directory):
        self._directory = directory
        if not os.path.isdir(directory):
            os.makedirs(directory)


class Dataset(object):
    """ A generic dataset.

    Parameters/Attributes
    ---------------------
    X: any
        Input data.
    y: any
        Output data.

    """
    def __init__(self, X, y=None):
        if len(X) == 1:
            X = [X]
        self.X = X

        self.unsup = y is None

        if not self.unsup:
            if len(y) == 1:
                y = [y]
            self.y = y
            assert len(self.X) == len(self.y)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __str__(self):
        return "<Dataset. len=%d>" % len(self)

    def __repr__(self):
        return str(self)

    def __iter__(self):
        return self.X if self.unsup else zip(self.X, self.y)

    def __getitem__(self, key):
        return Dataset(
            X=self.X[key],
            y=(None if self.unsup else self.y[key]))


class ExperimentDirectory(object):
    """ A directory for storing data from an experiment. """

    def __init__(self, directory, exp_dir):
        """
        Parameters
        ----------
        directory: str
            Name of the parent of the experiment directory.
        exp_dir: str
            Name of the experiment directory.

        """
        self.directory = directory
        self.exp_dir = exp_dir
        self._path = os.path.join(directory, self.exp_dir)

        assert os.path.isdir(self._path), (
            "Creating an ExperimentDirectory object, but the "
            "corresponding directory on the filesystem (%s) "
            "does not exist." % self._path)

    def __str__(self):
        return "ExperimentDirectory(%s, %s)" % (self.directory, self.exp_dir)

    def __repr__(self):
        return str(self)

    def path_for(self, path, is_dir=False):
        """ Get a path for a file, creating necessary subdirs. """
        if is_dir:
            filename = ""
        else:
            path, filename = os.path.split(path)

        full_path = os.path.join(self._path, path)
        if not os.path.isdir(full_path):
            os.makedirs(full_path)
        return os.path.join(full_path, filename)

    @staticmethod
    def create_new(directory, exp_name, data=None, use_time=False):
        """ Create a new experiment directory.

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
        exp_dir = make_filename(
            exp_name, use_time=use_time, config_dict=data)

        path = os.path.join(directory, exp_dir)
        os.makedirs(path)

        make_symlink(exp_dir, os.path.join(directory, 'latest'))

        return ExperimentDirectory(directory, exp_dir)


def get_latest_exp_dir(directory):
    latest = os.readlink(os.path.join(directory, 'latest'))
    return ExperimentDirectory(directory, latest)


def get_latest_results(directory, filename='results'):
    exp_dir = get_latest_exp_dir(directory)
    return pd.read_csv(exp_dir.path_for(filename))


def default_score(estimator, X, y=None):
    return estimator.score(X, y)


def get_n_points(dists):
    """ Get the number of points in a dictionary designed for use
        as the ``param_distributions`` argument to RandomizedSearchCV.

    """
    if not dists:
        return 1

    n_points = 1
    for k, v in six.iteritems(dists):
        if hasattr(v, 'rvs'):
            return np.inf
        else:
            assert hasattr(v, '__len__')
            n_points *= len(v)
    return n_points


def save_object(filename, obj):
    with open(filename, 'wb') as f:
        dill.dump(obj, f, protocol=dill.HIGHEST_PROTOCOL)


class Experiment(object):
    """ Run machine learning experiment.

        Two modes are available:
            data: Explores how the performance of one or more
                estimators changes as some parameter of the *data
                generation process* is varied.

            estimator: Explores how the performance of one or more
                estimators changes as some parameter of the *estimators*
                is varied (tested estimators need to accept the varied
                parameter in their __init__).

        Results are saved as a csv file after running each estimator
        on a new dataset. So even if execution is terminated part-way through
        the experiment, all results gathered so far will still be available.

        Parameters
        ----------
        mode: one of ('data', 'estimator')
            Controls whether the x-variable is supplied to the data
            generation process or is used as an attribute on the estimators.
        base_estimators: instance sklearn.Estimator or list of such instances
            The base_estimators to be tested. The Estimator classes must
            provide the method ``point_distribution``.
        x_var_name: string
            Name to use for the x variable (aka independent variable).
        x_var_values: list-like
            Values that the x variable can take on.
        generate_data: function
            A function which generates data. Must accept an argument with
            name equal to ``x_var_name``. Must return either a train and
            test set, or a train set, test set, and dictionary giving
            context/annotation information on the generated data. If this
            context dictionary contains the key `true_model`, then the value
            at this key will also have the test score functions applied to it.
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
        directory: str (optional)
            Name of an "experiments" directory. A new directory inside
            ``exp_dir`` will be created for this experiment, and relevant
            data will be stored in there (experimental results, plots, etc).
        save_estimators: bool (optional, default=False)
            Whether to store estimators (uses cPickle).
        save_datasets: bool (optional, default=False)
            Whether to store datasets (uses cPickle).
        name: str (optional)
            Name for the experiment.
        exp_dir: ExperimentDirectory instance (optional)
            Instead of creating a new directory for the experiment,
            use an existing one.
        params: dict (optional)
            A dictionary of parameters to be written to file, helping to
            identify the conditions that the experiment was run under.

    """
    def __init__(
            self, mode, base_estimators, x_var_name, x_var_values,
            generate_data, score=None, n_repeats=5, data_kwargs=None,
            search_kwargs=None, directory='.', save_estimators=False,
            save_datasets=False, name='experiment', exp_dir=None,
            params=None):
        self._constructor_params = locals().copy()
        del self._constructor_params['self']

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
                    score_func = default_score if s[0] is None else s[0]
                    score_name = s[1]
                else:
                    score_func = default_score if s is None else s
                    score_name = (
                        s.__name__ if hasattr(s, "__name__") else str(s))
                    score_name = (score_name if score_name.startswith('score')
                                  else 'score_' + score_name)

                self.scores.append(score_func)
                self.score_names.append(score_name)
        else:
            self.scores = [default_score]
            self.score_names = ['default_score']
        self.score = self.scores[0]

        self.n_repeats = n_repeats
        self.data_kwargs = {} if data_kwargs is None else data_kwargs

        self.search_kwargs = dict(
            n_iter=10, n_jobs=1, cv=None, iid=False,
            error_score=np.inf, pre_dispatch='n_jobs')
        if search_kwargs is not None:
            self.search_kwargs.update(search_kwargs)

        if exp_dir is None:
            exp_dir = ExperimentDirectory.create_new(
                directory, name, use_time=True)
        self.exp_dir = exp_dir

        self.save_estimators = save_estimators
        self.save_datasets = save_datasets

        self.params = params

        try:
            self.script_path = os.path.abspath(
                sys.modules['__main__'].__file__)
        except AttributeError:
            self.script_path = "None"
        self._constructor_params['script_path'] = self.script_path

        with open(self.exp_dir.path_for('parameters'), 'w') as f:
            f.write(str(self))

    def __str__(self):
        s = "Experiment(\n"
        for k, v in six.iteritems(self._constructor_params):
            s += indent("%s=%s,\n" % (k, pprint.pformat(v)), 1)
        s += ")"
        return s

    def __repr__(self):
        return str(self)

    def run(self, filename='results', random_state=None):
        """ Run the experiment.

        Parameters
        ----------
        filename: str
            Name of the file to write results to.
        random_state: int seed, RandomState instance, or None (default)
            Random state for the experiment.

        """
        rng = check_random_state(random_state)
        estimator_rng = check_random_state(sd.gen_seed(rng))
        data_rng = check_random_state(sd.gen_seed(rng))
        search_rng = check_random_state(sd.gen_seed(rng))

        self.search_kwargs.update(dict(random_state=search_rng))

        self.df = None
        searches = defaultdict(list)

        handler = logging.FileHandler(
            filename=self.exp_dir.path_for("log.txt"), mode='w')
        handler.setFormatter(logging.Formatter())
        handler.setLevel(logging.DEBUG)
        logging.getLogger('').addHandler(handler)  # add to root logger

        for x in self.x_var_values:
            for r in range(self.n_repeats):
                results = []

                train, test, context = self._draw_dataset(x, r, data_rng)

                if context is not None and 'true_model' in context:
                    logger.info("Testing ground-truth model...")

                    results.append({
                        'round': r, self.x_var_name: x,
                        'method': context['true_model'].name})

                    for s, sn in zip(self.scores, self.score_names):
                        score = s(context['true_model'], test.X, test.y)
                        logger.info("    Test score %s: %f." % (sn, score))
                        results[-1][sn] = score

                for base_est in self.base_estimators:
                    _results, _search = self._train_and_test(
                        base_est, x, r, train, test, context,
                        search_rng, estimator_rng)
                    searches[base_est.name].append(_search)
                    results.append(_results)

                # Save a snapshot of the results so far.
                if self.df is None:
                    self.df = pd.DataFrame.from_records(results)
                else:
                    self.df = self.df.append(results, ignore_index=True)
                self.df.to_csv(self.exp_dir.path_for(filename))

        logging.getLogger('').removeHandler(handler)

        self.searches = searches

        return self.df

    def _draw_dataset(self, x, r, data_rng):
        logger.info(as_title("Drawing new dataset...   "))
        data_kwargs = self.data_kwargs.copy()
        data_kwargs['random_state'] = sd.gen_seed(data_rng)
        if self.mode == 'data':
            data_kwargs[self.x_var_name] = x

        data = self.generate_data(**data_kwargs)
        assert isinstance(data, tuple) and (
            len(data) == 2 or len(data) == 3)

        if self.save_datasets:
            filename = (
                'datasets/{x_var_name}={x_var_value}_round={round}'.format(
                    x_var_name=self.x_var_name,
                    x_var_value=x, round=r))
            filepath = self.exp_dir.path_for(filename)
            save_object(filepath, data)

        if len(data) == 2:
            train, test = data
            context = None
        else:
            train, test, context = data

        return train, test, context

    def _train_and_test(
            self, base_est, x, r, train, test, context,
            search_rng, estimator_rng):

        est = sklearn.base.clone(base_est)
        est_seed = sd.gen_seed(estimator_rng)
        est.random_state = est_seed

        dirname = (
            'estimators/{est_name}_'
            '{x_var_name}={x_var_value}_'
            'round={round}'.format(
                est_name=est.name, x_var_name=self.x_var_name,
                x_var_value=x, round=r))
        dirpath = self.exp_dir.path_for(dirname, is_dir=True)

        if 'directory' in est.get_params():
            est.directory = dirpath

        est_params = est.get_params()

        if self.mode == 'estimator':
            if self.x_var_name not in est_params:
                raise ValueError(
                    "Estimator %s does not accept a parameter "
                    "called %s." % (est, self.x_var_name))
            setattr(est, self.x_var_name, x)

        logger.info(
            "Collecting data point. "
            "method: %s, %s: %s, repeat: %d, seed: %d."
            "" % (base_est.name, self.x_var_name,
                  str(x), r, est_seed))

        try:
            dists = est.point_distribution(context=context)
        except KeyError as e:
            raise KeyError(
                "Key {key} required by estimator {est} was "
                "not in context {context} supplied by the data "
                "generation function.".format(
                    key=e, est=est, context=context))

        for name, dist in six.iteritems(dists):
            if hasattr(dist, 'rvs'):
                dist.random_state = search_rng

        # Make sure we don't CV over the independent variable.
        if self.mode == 'estimator':
            dists.pop(self.x_var_name, None)

        for key in dists:
            if key not in est_params:
                raise ValueError(
                    "Point distribution contains field "
                    "``%s`` which is not a parameter of "
                    "the estimator %s." % (key, est))

        n_points = get_n_points(dists)
        n_iter = min(
            self.search_kwargs.get('n_iter', 10), n_points)

        if n_points > n_iter:
            search = RandomizedSearchCV(
                est, dists, scoring=self.scores[0],
                **self.search_kwargs)
            logger.info("    Running RandomizedSearchCV for "
                        "%d iters..." % n_iter)
        else:
            # These are arguments for RandomizedSearchCV
            # but not GridSearchCV
            _search_kwargs = self.search_kwargs.copy()
            _search_kwargs.pop('random_state', None)
            _search_kwargs.pop('n_iter', None)

            search = GridSearchCV(
                est, dists, scoring=self.scores[0],
                **_search_kwargs)
            logger.info(
                "    Running GridSearchCV for "
                "%d iters..." % n_iter)

        then = time.time()
        search.fit(train.X, train.y)
        search_time = time.time() - then

        logger.info("    Best params: %s." % search.best_params_)
        logger.info("    Search time: "
                    "%s seconds per point." % (search_time/n_iter))

        learned_est = search.best_estimator_

        if self.save_estimators:
            estpath = os.path.join(dirpath, 'estimator')
            save_object(estpath, learned_est)

        results = {
            'round': r, self.x_var_name: x,
            'method': base_est.name,
            'search_time_per_point': search_time/n_iter}

        best_grid_score = max(
            search.grid_scores_,
            key=lambda gs: gs.mean_validation_score)

        train_score = best_grid_score.mean_validation_score
        train_score_std = np.std(
            best_grid_score.cv_validation_scores)

        logger.info("    Training score mean: %f." % train_score)
        results['training_score'] = train_score

        logger.info(
            "    Training score std: %f." % train_score_std)
        results['training_score_std'] = train_score_std

        record_attrs = (
            [] if not hasattr(learned_est, 'record_attrs')
            else learned_est.record_attrs)

        for attr in record_attrs:
            value = getattr(learned_est, attr)
            try:
                value = float(value)
            except ValueError:
                pass

            results[attr] = value
            logger.info(
                "    Value for attr %s: %s." % (attr, value))

        logger.info("    Running tests...")
        then = time.time()
        for s, sn in zip(self.scores, self.score_names):
            score = s(learned_est, test.X, test.y)
            logger.info("    Test score %s: %f." % (sn, score))
            results[sn] = score
        test_time = time.time() - then
        logger.info("    Test time: %s seconds." % test_time)
        results['test_time'] = test_time

        return results, search


ExperimentSpec = namedtuple('ExperimentSpec', 'title score score_display '
                            'x_var_name x_var_values '
                            'x_var_display n_repeats kwargs')


def run_experiment_and_plot(
        mode, estimators, data_generator, specs, quick_specs,
        data_kwargs, spec_names='all', search_kwargs=None,
        directory='/data/experiments', random_state=None,
        labels=None, params=None):
    """ A utility function which handles much of the boilerplate required to set
        up a script running a set of experiments and plotting the results.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--plot', action='store_true',
        help='If supplied, will plot the most recent set of experiments '
             'rather running a new set.')
    parser.add_argument(
        '--time', action='store_true',
        help='If supplied, will plot timing '
             'information as well as score information.')
    parser.add_argument(
        '--show', action='store_true',
        help='If supplied, displays the plots as they are created.')
    parser.add_argument(
        '--quick', action='store_true',
        help='If supplied, run a set of small test experiments.')
    parser.add_argument(
        '--n-jobs', type=int, default=1,
        help="Number of jobs to use for cross validation.")
    parser.add_argument(
        '-r', action='store_true',
        help="If supplied, exceptions encountered during "
             "cross-validation will propagated all the way up. Otherwise, "
             "the exceptions are caught and the candidate parameter setting "
             "receives a score of negative infinity.")

    logging.basicConfig(level=logging.INFO, format='')

    args = parser.parse_args()
    if args.quick:
        specs = quick_specs

    labels = {} if labels is None else labels
    search_kwargs = {} if search_kwargs is None else search_kwargs
    search_kwargs.update(
        n_jobs=args.n_jobs,
        error_score='raise' if args.r else -np.inf)

    if spec_names == 'all':
        spec_names = specs.keys()
    else:
        spec_names = spec_names.split(' ')

    experiments = []
    figs = []

    for spec_name in spec_names:
        print("Running experiment %s." % spec_name)

        spec = specs[spec_name]
        title = spec.title
        score = spec.score
        score_display = spec.score_display
        x_var_name = spec.x_var_name
        x_var_values = spec.x_var_values
        x_var_display = spec.x_var_display
        n_repeats = spec.n_repeats

        _data_kwargs = data_kwargs.copy()
        _data_kwargs.update(spec.kwargs)

        experiment_name = "%s:%s" % (spec_name, x_var_name)

        if not args.plot:
            experiment = Experiment(
                mode, estimators, x_var_name, x_var_values,
                data_generator, score,
                n_repeats=n_repeats, data_kwargs=_data_kwargs,
                search_kwargs=search_kwargs, directory=directory,
                name=experiment_name, params=params)

            df = experiment.run(
                'results_%s.csv' % experiment_name, random_state=random_state)
            exp_dir = experiment.exp_dir
            with open(exp_dir.path_for('experiment.pkl'), 'wb') as f:
                dill.dump(experiment, f, protocol=dill.HIGHEST_PROTOCOL)
        else:
            exp_dir = get_latest_exp_dir(directory)
            df = pd.read_csv(
                exp_dir.path_for('results_%s.csv' % experiment_name))
            try:
                with open(exp_dir.path_for('experiment.pkl'), 'rb') as f:
                    experiment = dill.load(f)
            except:
                experiment = None

        experiments.append(experiment)

        markers = MarkerStyle.filled_markers
        colors = seaborn.color_palette("hls", 16)

        def plot_kwarg_func(sv):
            idx = hash(str(sv))
            marker = markers[idx % len(markers)]
            c = colors[idx % len(colors)]
            label = labels.get(sv, sv)
            return dict(label=label, c=c, marker=marker)

        measure_names = (
            experiment.score_names if experiment is not None else
            [mn for mn in df.columns if 'score' in mn])

        if args.time:
            measure_names += [mn for mn in df.columns if 'time' in mn]

        plt.figure(figsize=(10, 5))
        fig, axes = plot_measures(
            df, measure_names, x_var_name, 'method',
            legend_outside=True,
            kwarg_func=plot_kwarg_func,
            measure_display=score_display,
            x_var_display=x_var_display)
        axes[0].set_title(title)

        # plt.rc('text', usetex=True)
        plt.rc('font', family='serif', size=22)

        fig.subplots_adjust(left=0.1, right=0.85, top=0.94, bottom=0.12)

        plt.savefig(exp_dir.path_for('plot_%s.pdf' % x_var_name))

        if args.show:
            plt.show()

        figs.append(fig)

    return experiments, figs
