import pprint
import logging
import six
import os
from os.path import join as pjoin
import time
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import sys
import shutil
import seaborn.apionly as sns
import traceback
import copy
import dill as pickle

from sklearn.base import clone
from sklearn.utils import check_random_state
import collections
try:
    from sklearn.model_selection import (
        RandomizedSearchCV, GridSearchCV, ParameterGrid, ParameterSampler, cross_val_score)
except ImportError:
    from sklearn.grid_search import (
        RandomizedSearchCV, GridSearchCV, ParameterGrid, ParameterSampler, cross_val_score)

import spectral_dagger as sd
from spectral_dagger.utils.plot import plot_measures
from spectral_dagger.utils.misc import (
    make_symlink, make_filename, indent, as_title, send_email_using_cfg, ObjectLoader, ObjectSaver)


pp = pprint.PrettyPrinter()
verbosity = 2

logger = logging.getLogger(__name__)


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
        return self.X if self.unsup else list(zip(self.X, self.y))

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
        self._path = pjoin(directory, self.exp_dir)

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

        full_path = pjoin(self._path, path)
        if not os.path.isdir(full_path):
            os.makedirs(full_path)
        return pjoin(full_path, filename)

    def move_to_directory(self, new_directory):
        if not os.path.isdir(new_directory):
            raise ValueError("%s is not a directory." % new_directory)

        shutil.move(self._path, new_directory)
        self.directory = new_directory
        self._path = pjoin(new_directory, self.exp_dir)

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

        _path = pjoin(directory, exp_dir)
        os.makedirs(_path)

        make_symlink(exp_dir, pjoin(directory, 'latest'))

        return ExperimentDirectory(directory, exp_dir)


def get_latest_exp_dir(directory):
    latest = os.readlink(pjoin(directory, 'latest'))
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
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def set_nested_value(d, key, value, sep="."):
    """ Set value in a nested dictionary.

    Parameters
    ----------
    d: dict
        Dictionary to modify. It is also returned.
    key: str
        Hierarchical key representing the value to modify. Levels in the key
        are separated by occurrences of ``sep``. If the nested dictionaries
        required to add a value at the key do not exist, they are created.
    value: any
        Value to place at key.
    sep: str
        String used to separate levels in the key.

    """
    assert isinstance(d, dict)
    dct = d
    keys = key.split(sep)
    for k in keys[:-1]:
        current_value = dct.get(k, None)

        if current_value is None:
            dct[k] = {}
            dct = dct[k]
        else:
            if not isinstance(dct[k], dict):
                raise Exception(
                    "Value at '%s' in %s should be a dictionary, "
                    "has value %s instead." % (k, dct, dct[k]))
            dct = dct[k]

    dct[keys[-1]] = value
    return d


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
            Name of the x variable (aka independent variable). Must be
            equal to the name that will be used when passing the x variable
            into either the fit method of the estimators (when mode='estimator'),
            or the function generate_data. This can also be a hierarchical name,
            with periods separating the levels, when one of the arguments is a
            (possibly heavily nested) dictionary. The top-level name specifies
            the argument name, and each successive name specifies a key in a nested
            dictionary.
        x_var_values: list-like
            Values that the x variable can take on.
        generate_data: function
            A function which generates data. When mode='data', must accept an
            argument with name equal to ``x_var_name``. Must return either a
            train and test set, or a train set, test set, and dictionary giving
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
        hook: f(experiment, dataframe) -> None (optional)
            A hook called after all the work for one x-value is completed. Can,
            for instance, be used to plot or take a snapshot of progress so far.
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
            save_datasets=False, hook=None, name=None, exp_dir=None,
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
            if isinstance(score, tuple) or isinstance(score, collections.Callable):
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

        if name is None:
            script_path = os.path.abspath(
                sys.modules['__main__'].__file__)
            name = os.path.splitext(os.path.basename(script_path))[0]
        self.name = name

        if exp_dir is None:
            exp_dir = ExperimentDirectory.create_new(
                directory, name, use_time=True)
        self.directory = directory
        self.exp_dir = exp_dir

        self.save_estimators = save_estimators
        self.save_datasets = save_datasets
        self.hook = hook

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
        s = "%s(\n" % self.__class__.__name__
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
        print("Running experiment: %s." % self.name)

        complete = False

        try:
            rng = check_random_state(random_state)
            estimator_rng = check_random_state(sd.gen_seed(rng))
            data_rng = check_random_state(sd.gen_seed(rng))
            search_rng = check_random_state(sd.gen_seed(rng))

            self.search_kwargs.update(dict(random_state=search_rng))

            self.df = None

            handler = logging.FileHandler(
                filename=self.exp_dir.path_for("log.txt"), mode='w')
            handler.setFormatter(logging.Formatter())
            handler.setLevel(logging.DEBUG)
            logging.getLogger('').addHandler(handler)  # add to root logger

            estimator_names = [base_est.name for base_est in self.base_estimators]
            if len(set(estimator_names)) < len(estimator_names):
                raise Exception(
                    "Multiple base estimators have the same name. "
                    "Names are: %s" % estimator_names)

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
                        _results = self._train_and_test(
                            base_est, x, r, train, test, context,
                            search_rng, estimator_rng)
                        results.append(_results)

                    # Save a snapshot of the results so far.
                    if self.df is None:
                        self.df = pd.DataFrame.from_records(results)
                    else:
                        self.df = self.df.append(results, ignore_index=True)
                    self.df.to_csv(self.exp_dir.path_for(filename))

                if self.hook is not None:
                    self.hook(self, self.df)

            logging.getLogger('').removeHandler(handler)
            complete = True
        finally:
            # Move the experiment to a folder based on
            # whether it completed successfully.
            new_loc = 'complete' if complete else 'incomplete'
            new_path = pjoin(self.directory, new_loc)
            try:
                os.makedirs(new_path)
            except:
                pass
            self.exp_dir.move_to_directory(new_path)
            make_symlink(
                self.exp_dir.exp_dir, pjoin(new_path, 'latest'))

        return self.df

    def _draw_dataset(self, x, r, data_rng):
        logger.info(as_title("Drawing new dataset...   "))

        data_kwargs = copy.deepcopy(self.data_kwargs)
        data_kwargs.update(random_state=sd.gen_seed(data_rng))
        if self.mode == 'data':
            set_nested_value(data_kwargs, self.x_var_name, x, sep='__')

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

        est = clone(base_est)
        if self.mode == 'estimator':
            est.set_params(**{self.x_var_name: x})

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

        # We can only do this check at the top level of params.
        for key in dists:
            if key not in est.get_params(deep=True):
                raise ValueError(
                    "Point distribution contains field "
                    "``%s`` which is not a parameter of "
                    "the estimator %s." % (key, est))

        n_points = get_n_points(dists)
        n_iter = min(self.search_kwargs.get('n_iter', 10), n_points)

        if n_iter == 1 or n_iter == 0:
            logger.info("    Only one candidate point, skipping cross-validation.")
            search_time_pp = 0.0
            train_score = 0.0
            train_score_std = 0.0

            if n_iter == 1:
                est.set_params(**{k: v[0] for k, v in six.iteritems(dists)})
            learned_est = est

            then = time.time()
            learned_est.fit(train.X, train.y)
            logger.info("    Fit time: %s seconds." % (time.time() - then))
        else:
            if n_points > n_iter:
                search = RandomizedSearchCV(
                    est, dists, scoring=self.scores[0], **self.search_kwargs)
                logger.info("    Running RandomizedSearchCV for "
                            "%d iters..." % n_iter)
            else:
                # These are arguments for RandomizedSearchCV
                # but not GridSearchCV
                _search_kwargs = self.search_kwargs.copy()
                _search_kwargs.pop('random_state', None)
                _search_kwargs.pop('n_iter', None)

                search = GridSearchCV(
                    est, dists, scoring=self.scores[0], **_search_kwargs)
                logger.info(
                    "    Running GridSearchCV for "
                    "%d iters..." % n_iter)

            then = time.time()
            search.fit(train.X, train.y)
            search_time_pp = (time.time() - then) / n_iter

            best_grid_score = max(
                search.grid_scores_,
                key=lambda gs: gs.mean_validation_score)

            train_score = best_grid_score.mean_validation_score
            train_score_std = np.std(best_grid_score.cv_validation_scores)

            learned_est = search.best_estimator_

            logger.info("    Best params: %s." % search.best_params_)
            logger.info("    Search time: "
                        "%s seconds per point." % search_time_pp)

        if self.save_estimators:
            estpath = pjoin(dirpath, 'estimator')
            save_object(estpath, learned_est)

        results = {
            'round': r, self.x_var_name: x,
            'method': base_est.name,
            'search_time_per_point': search_time_pp}

        logger.info("    Training score mean: %f." % train_score)
        results['training_score'] = train_score

        logger.info(
            "    Training score std: %f." % train_score_std)
        results['training_score_std'] = train_score_std

        for attr in est.record_attrs:
            try:
                value = getattr(learned_est, attr)
            except AttributeError:
                value = learned_est.get_params(deep=True)[attr]

            try:
                value = float(value)
            except (ValueError, TypeError):
                pass
            results[attr] = value
            logger.info("    Value for attr %s: %s." % (attr, value))

        logger.info("    Running tests...")
        then = time.time()
        for s, sn in zip(self.scores, self.score_names):
            score = s(learned_est, test.X, test.y)
            logger.info("    Test score %s: %f." % (sn, score))
            results[sn] = score
        test_time = time.time() - then
        logger.info("    Test time: %s seconds." % test_time)
        results['test_time'] = test_time

        return results


def _plot(
        experiment, df, plot_path, legend_loc='right',
        x_var_display="", score_display=None, title="", labels=None, show=False):
    return __plot(
        experiment.score_names, experiment.x_var_name, df, plot_path, legend_loc,
        x_var_display, score_display, title, labels, show)


def __plot(
        score_names, x_var_name, df, plot_path, legend_loc='right',
        x_var_display="", score_display=None, title="", labels=None, show=False):

    if os.path.isfile(plot_path):
        os.rename(plot_path, plot_path + '.bk')

    markers = MarkerStyle.filled_markers
    colors = sns.color_palette("hls", 16)
    labels = {} if labels is None else labels

    def plot_kwarg_func(sv):
        idx = hash(str(sv))
        marker = markers[idx % len(markers)]
        c = colors[idx % len(colors)]
        label = labels.get(sv, sv)
        return dict(label=label, c=c, marker=marker)

    measure_names = score_names
    if time:
        measure_names = measure_names + [mn for mn in df.columns if 'time' in mn]

    if not x_var_display:
        x_var_display = x_var_name

    plt.figure(figsize=(10, 5))
    fig, axes = plot_measures(
        df, measure_names, x_var_name, 'method',
        legend_loc=legend_loc,
        kwarg_func=plot_kwarg_func,
        measure_display=score_display,
        x_var_display=x_var_display)
    axes[0].set_title(title)

    # plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=10)

    fig.subplots_adjust(left=0.1, right=0.85, top=0.94, bottom=0.12)

    plt.savefig(plot_path)

    if show:
        plt.show()
    return fig


def make_plot_hook(plot_name, **plot_kwargs):
    def plot_hook(experiment, df):
        plot_path = experiment.exp_dir.path_for(plot_name)
        _plot(experiment, df, plot_path, **plot_kwargs)
    return plot_hook


def _finish(experiment, email_cfg, success, **plot_kwargs):
    exp_dir = experiment.exp_dir
    save_object(exp_dir.path_for('experiment.pkl'), experiment)
    plot_path = exp_dir.path_for('plot.pdf')

    fig = None
    if experiment.df is not None:
        fig = _plot(experiment, experiment.df, plot_path, **plot_kwargs)

    if email_cfg:
        if success:
            subject = "Experiment %s complete!" % experiment.name
            body = ''
        else:
            subject = "Experiment %s failed." % experiment.name
            body = ''.join(traceback.format_exception(*sys.exc_info()))
        send_email_using_cfg(
            email_cfg, subject=subject, body=body, files_to_attach=plot_path)
    return fig


def run_experiment_and_plot(
        exp_kwargs, quick_exp_kwargs=None, random_state=None, **plot_kwargs):
    """ A utility function which handles much of the boilerplate required to set
        up a script running a set of experiments and plotting the results.

        plot_kwargs: Extra arguments passed to the _plot function.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--seed', type=int, default=None,
        help='If supplied, will seed the random number generator with this integer.')
    parser.add_argument(
        '--plot', action='store_true',
        help='If supplied, will plot the most recent completed '
             'experiment rather running a new experiment.')
    parser.add_argument(
        '--plot-incomplete', action='store_true',
        help='If supplied, will plot the most recent non-completed '
             'experiment instead of running a new experiment.')
    parser.add_argument(
        '--name', type=str,
        help='Name of the experiment. If not supplied, the name of the '
             'calling script will be used.', default=None)
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
        '--parallel', action='store_true',
        help='If supplied, build the experiment to be executed in parallel.')
    parser.add_argument(
        '--n-jobs', type=int, default=1,
        help="Number of jobs to use for cross validation.")
    parser.add_argument(
        '--save-est', action='store_true',
        help="If supplied, trained estimators will be saved.")
    parser.add_argument(
        '--save-data', action='store_true',
        help="If supplied, data sets will be saved.")
    parser.add_argument(
        '--email-cfg', type=str, default=None,
        help="Name of a file from which to pull email configurations. "
             "If not provided, no email will be sent. If provided, email "
             "will be sent when experiment completes, along with a specification of "
             "the error if the experiment failed, or the created plot if the "
             "experiment completed successfully.")
    parser.add_argument(
        '-r', action='store_true',
        help="If supplied, exceptions encountered during "
             "cross-validation will propagated all the way up. Otherwise, "
             "such exceptions are caught and the candidate parameter setting "
             "receives a score of negative infinity.")
    args, _ = parser.parse_known_args()

    plot_kwargs['show'] = args.show or args.plot or args.plot_incomplete

    random_state = check_random_state(random_state)
    if args.seed is not None:
        random_state.seed(args.seed)

    # Setup and run experiment
    logging.basicConfig(level=logging.INFO, format='')

    if args.quick and quick_exp_kwargs is not None:
        exp_kwargs = quick_exp_kwargs

    exp_kwargs['name'] = args.name or None
    exp_kwargs['save_estimators'] = args.save_est
    exp_kwargs['save_datasets'] = args.save_data

    if 'search_kwargs' not in exp_kwargs:
        exp_kwargs['search_kwargs'] = {}

    exp_kwargs['search_kwargs'].update(
        n_jobs=args.n_jobs,
        error_score='raise' if args.r else -np.inf)

    if 'directory' not in exp_kwargs:
        exp_kwargs['directory'] = '/data/experiment'

    plot_kwargs['title'] = plot_kwargs.get('title', exp_kwargs['name'])
    exp_kwargs['hook'] = make_plot_hook('plot.pdf', **plot_kwargs)

    if args.parallel:
        experiment = ParallelExperiment(**exp_kwargs)
        experiment.build()

        archive_name = experiment.exp_dir.exp_dir
        print("Zipping built experiment as {}.zip.".format(archive_name))
        shutil.make_archive(archive_name, 'zip', *os.path.split(experiment.exp_dir._path))

        print("Experiment has been built, execute using the ``sd-experiment`` command-line utility.")
        return

    if args.plot or args.plot_incomplete:
        # Re-plotting an experiment that has already been run.
        exp_dir = get_latest_exp_dir(
            pjoin(
                exp_kwargs['directory'],
                'complete' if args.plot else 'incomplete'))

        plot_path = exp_dir.path_for('plot.pdf')

        df = pd.read_csv(exp_dir.path_for('results.csv'))

        with open(exp_dir.path_for('experiment.pkl'), 'rb') as f:
            experiment = pickle.load(f)

        fig = _plot(experiment, df, plot_path, **plot_kwargs)
    else:
        # Running a new experiment and then plotting.
        experiment = Experiment(**exp_kwargs)

        try:
            experiment.run('results.csv', random_state=random_state)
        except:
            exc_info = sys.exc_info()
            try:
                fig = _finish(experiment, args.email_cfg, False, **plot_kwargs)
            except:
                print("Error while cleaning up:")
                traceback.print_exc()
            raise exc_info[0](exc_info[1]).with_traceback(exc_info[2])

        fig = _finish(experiment, args.email_cfg, True, **plot_kwargs)

    return experiment, fig


class ParallelExperiment(Experiment):
    def build(self, random_state=None):
        """
        Parameters
        ----------
        filename: str
            Name of the file to write results to.
        random_state: int seed, RandomState instance, or None (default)
            Random state for the experiment.

        """
        print(
            "Building parallel experiment file for experiment: {} "
            "in directory: {}.".format(self.name, self.exp_dir.exp_dir))
        self.saver = ObjectSaver(self.exp_dir._path, eager=True)

        rng = check_random_state(random_state)
        data_rng = check_random_state(sd.gen_seed(rng))
        search_rng = check_random_state(sd.gen_seed(rng))

        self.search_kwargs.update(dict(random_state=search_rng))

        handler = logging.FileHandler(
            filename=self.exp_dir.path_for("log.txt"), mode='w')
        handler.setFormatter(logging.Formatter())
        handler.setLevel(logging.DEBUG)
        logging.getLogger('').addHandler(handler)  # add to root logger

        estimator_names = [base_est.name for base_est in self.base_estimators]
        if len(set(estimator_names)) < len(estimator_names):
            raise Exception(
                "Multiple base estimators have the same name. "
                "Names are: %s" % estimator_names)

        self.results = {}  # test_scenario_idx -> results

        for x in self.x_var_values:
            for r in range(self.n_repeats):
                train, test, context = self._draw_dataset(x, r, data_rng)

                train_idx = self.saver.add_object(train, 'train', context=context)
                test_idx = self.saver.add_object(test, 'test', context=context)
                assert train_idx == test_idx

                for base_est in self.base_estimators:
                    est = clone(base_est)
                    if self.mode == 'estimator':
                        est.set_params(**{self.x_var_name: x})

                    train_indices = self._build_train_scenarios(
                        base_est, x, r, train_idx, context, search_rng)

                    self._build_test_scenario(
                        base_est, x, r, test_idx, train_indices, context)

        logging.getLogger('').removeHandler(handler)

        results = {
            'results': self.results, 'x_var_name': self.x_var_name, 'score_names': self.score_names}

        with open(self.exp_dir.path_for('_results.pkl'), 'w') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

        print("Created {} train scenarios.".format(self.saver.n_objects('train_scenario')))
        print("Created {} test scenarios.".format(self.saver.n_objects('test_scenario')))

    def _build_train_scenarios(self, est, x, r, train_idx, context, search_rng):
        logger.info(
            "Saving data point. "
            "method: %s, %s: %s, repeat: %d."
            "" % (est.name, self.x_var_name, str(x), r))

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
            if key not in est.get_params(deep=True):
                raise ValueError(
                    "Point distribution contains field "
                    "``%s`` which is not a parameter of "
                    "the estimator %s." % (key, est))

        n_points = get_n_points(dists)
        n_iter = min(self.search_kwargs.get('n_iter', 10), n_points)

        if n_iter == 1 or n_iter == 0:
            logger.info("    Only one candidate point, skipping cross-validation.")
        else:
            if n_points > n_iter:
                search = RandomizedSearchCV(
                    est, dists, scoring=self.scores[0], **self.search_kwargs)

                param_iter = ParameterSampler(search.param_distributions,
                                              search.n_iter,
                                              random_state=search.random_state)
                logger.info("    Using RandomizedSearchCV with %d iters..." % n_iter)

            else:
                # These are arguments for RandomizedSearchCV but not GridSearchCV
                _search_kwargs = self.search_kwargs.copy()
                _search_kwargs.pop('random_state', None)
                _search_kwargs.pop('n_iter', None)

                search = GridSearchCV(est, dists, scoring=self.scores[0], **_search_kwargs)
                param_iter = ParameterGrid(search.param_grid)
                logger.info("    Using GridSearchCV with %d iters..." % n_iter)

        cv = self.search_kwargs.get('cv', None)

        train_indices = {}
        for params in param_iter:
            cloned_est = clone(est).set_params(**params)

            ti = self.saver.add_object(
                cloned_est, 'train_scenario', scoring=self.scores[-1],
                cv=cv, dataset_idx=train_idx)

            train_indices[ti] = params

        return train_indices

    def _build_test_scenario(self, est, x, r, test_idx, train_indices, context):
        test_scenario_idx = self.saver.add_object(
            est, 'test_scenario', scores=self.scores,
            score_names=self.score_names,
            train_indices=train_indices, dataset_idx=test_idx)

        self.results[test_scenario_idx] = {
            'round': r, self.x_var_name: x, 'method': est.name}


def scenario(scenario_type, creates):
    """ Scenarios check whether they need to run before they run.

    Parameter
    ---------
    creates: str or list of str
        Key that this scenario creates.

    """
    if isinstance(creates, str):
        creates = [creates]

    def f(w):
        def wrapper(directory, scenario_idx, verbose, force):
            loader = ObjectLoader(directory)

            finished = True
            for c in creates:
                try:
                    loader.load_object(c, scenario_idx)
                except KeyError:
                    finished = False
                    break

            if finished and not force:
                print("Skipping {} scenario {} since it has "
                      "already been run.".format(scenario_type, scenario_idx))
            else:
                w(directory, scenario_idx, verbose)

        return wrapper

    return f


@scenario('training', 'cv_score')
def run_training_scenario(directory, scenario_idx, verbose):
        if verbose:
            print("Beginning training scenario {}.".format(scenario_idx))

        loader = ObjectLoader(directory)

        estimator, kwargs = loader.load_object('train_scenario', scenario_idx)
        estimator.directory = pjoin(directory, 'train_scratch/{}'.format(scenario_idx))
        dataset_idx = kwargs.pop('dataset_idx')
        train, train_kwargs = loader.load_object('train', dataset_idx)

        scoring = kwargs.get('scoring', None)
        cv = kwargs.get('cv', None)

        cv_score = cross_val_score(
            estimator, train.X, train.y, scoring=scoring, cv=cv, n_jobs=1, verbose=0)

        saver = ObjectSaver(directory, eager=True)
        saver.add_object(cv_score, 'cv_score', scenario_idx)

        if verbose:
            print("Done training scenario {}.".format(scenario_idx))


def make_idx_print(idx):
    def idx_print(s):
        print("{}: {}".format(idx, s))
    return idx_print


@scenario('testing', 'test_scores')
def run_testing_scenario(directory, scenario_idx, verbose):
    # Read in the results of running, choose winner of CVs, test each.
    if verbose:
        print("Beginning testing scenario {}.".format(scenario_idx))

    loader = ObjectLoader(directory)

    estimator, kwargs = loader.load_object('test_scenario', scenario_idx)
    estimator.directory = pjoin(directory, 'test_scratch/{}'.format(scenario_idx))
    dataset_idx = kwargs.pop('dataset_idx')

    train_indices = kwargs.get('train_indices')
    scores = kwargs.get('scores', None)
    score_names = kwargs.get('score_names', None)

    cv_scores = {idx: np.mean(loader.load_object('cv_score', idx)[0]) for idx in train_indices}
    best_idx, best_cv_score = max(list(cv_scores.items()), key=lambda x: x[1])
    best_params = train_indices[best_idx]

    estimator.set_params(**best_params)

    train, train_kwargs = loader.load_object('train', dataset_idx)
    then = time.time()
    estimator.fit(train.X, train.y)

    idx_print = make_idx_print(scenario_idx)
    idx_print("Fit time: %s seconds." % (time.time() - then))

    test, test_kwargs = loader.load_object('test', dataset_idx)

    saver = ObjectSaver(directory, eager=True)

    idx_print("Running tests...")
    then = time.time()
    test_scores = {}
    for s, sn in zip(scores, score_names):
        score = s(estimator, test.X, test.y)
        idx_print("{},{},{}".format(scenario_idx, sn, score))
        test_scores[sn] = score

    test_time = time.time() - then
    idx_print("Test time: %s seconds." % test_time)

    saver.add_object(test_scores, 'test_scores', scenario_idx)

    if verbose:
        print("Done testing scenario {}.".format(scenario_idx))


def parallel_exp_plot(directory, **plot_kwargs):
    results_file = pjoin(directory, 'results.pkl')

    if os.path.exists(results_file):
        with open(pjoin(directory, 'results.pkl'), 'r') as f:
            results = pickle.load(f)
    else:
        with open(pjoin(directory, '_results.pkl'), 'r') as f:
            results = pickle.load(f)

        loader = ObjectLoader(directory)
        test_scores = loader.load_objects_of_kind('test_scores')

        for test_scenario_idx, (ts, _) in list(test_scores.items()):
            for sn, s in list(ts.items()):
                results['results'][test_scenario_idx][sn] = s

        with open(pjoin(directory, 'results.pkl'), 'w') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    score_names = results['score_names']
    x_var_name = results['x_var_name']

    df = pd.DataFrame.from_records(list(results['results'].values()))
    __plot(score_names, x_var_name, df, 'plot.pdf', **plot_kwargs)


def run_scenario(task, scenario_idx=-1, d='.', seed=None,
                 force=0, verbose=0, **kwargs):

    logging.basicConfig(level=logging.INFO, format='')

    if task == 'cv':
        run_training_scenario(d, scenario_idx, verbose, force)
    elif task == 'test':
        run_testing_scenario(d, scenario_idx, verbose, force)
    elif task == 'plot':
        parallel_exp_plot(d, **kwargs)
    else:
        raise ValueError("Unknown task {} for parallel experiment.".format(task))


def _run_scenario():
    from clify import command_line
    command_line(run_scenario, collect_kwargs=1, verbose=True)()
