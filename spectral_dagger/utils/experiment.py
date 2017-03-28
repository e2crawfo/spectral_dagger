import pprint
import logging
import six
import os
import time
import numpy as np
import pandas as pd
import sys
import shutil
import copy
import dill as pickle
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import seaborn.apionly as sns

from sklearn.base import clone
from sklearn.utils import check_random_state
import collections
try:
    from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
except ImportError:
    from sklearn.grid_search import RandomizedSearchCV, GridSearchCV

import spectral_dagger as sd
from spectral_dagger.utils.misc import make_symlink, make_filename, indent, as_title
from spectral_dagger.utils.plot import plot_measures

verbosity = 2
logger = logging.getLogger(__name__)
DEFAULT_EXPDIR = '/data/spectral_dagger/'


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


class ExperimentStore(object):
    """ Stores a collection of experiments. Each new experiment is assigned a fresh sub-path. """
    def __init__(self, path):
        self.path = os.path.abspath(path)
        try:
            os.makedirs(self.path)
        except:
            pass

    def new_experiment(self, name, data=None, use_time=False):
        """ Create a new experiment path. """
        filename = make_filename(name, use_time=use_time, config_dict=data)
        make_symlink(filename, os.path.join(self.path, 'latest'))
        return ExperimentDirectory(os.path.join(self.path, filename), store=self)

    def __str__(self):
        return "ExperimentStore(%s)".format(self.path)

    def __repr__(self):
        return str(self)

    def get_latest_experiment(self, kind=None):
        path = self.path
        if kind is not None:
            path = os.path.join(self.path, kind)

        latest = os.readlink(os.path.join(path, 'latest'))
        return ExperimentDirectory(latest)

    def get_latest_results(self, filename='results'):
        exp_dir = self.get_latest_exp_dir()
        return pd.read_csv(exp_dir.path_for(filename))

    def experiment_finished(self, exp_dir, success):
        dest_name = 'complete' if success else 'incomplete'
        dest_path = os.path.join(self.path, dest_name)
        try:
            os.makedirs(dest_path)
        except:
            pass
        shutil.move(exp_dir.path, dest_path)
        exp_dir.path = os.path.join(dest_path, os.path.basename(exp_dir.path))


class ExperimentDirectory(object):
    """ Wraps a directory storing data related to an experiment. """
    def __init__(self, path, store=None):
        self.path = path
        self.store = store

    def path_for(self, path, is_dir=False):
        """ Get a path for a file, creating necessary subdirs. """
        if is_dir:
            filename = ""
        else:
            path, filename = os.path.split(path)

        full_path = self.make_directory(path)
        return os.path.join(full_path, filename)

    def save_object(self, path, obj, mode='wb'):
        self.make_directory(path)

        with open(os.path.join(self.path, path) + '.pkl', mode) as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_object(self, path, obj, mode='rb'):
        with open(os.path.join(self.path, path) + '.pkl', mode) as f:
            obj = pickle.load(f)
        return obj

    def make_directory(self, path):
        full_path = os.path.join(self.path, path)
        try:
            os.makedirs(full_path)
        except:
            pass
        return full_path

    def finished(self, success):
        if self.store:
            self.store.experiment_finished(self, success)


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
        save_estimators: bool (optional, default=False)
            Whether to store estimators (uses cPickle).
        save_datasets: bool (optional, default=False)
            Whether to store datasets (uses cPickle).
        hook: f(experiment, dataframe) -> None (optional)
            A hook called after all the work for one x-value is completed. Can,
            for instance, be used to plot or take a snapshot of progress so far.
        name: str (optional)
            Name for the experiment.
        directory: str (optional)
            Name of directory for storing experiments. A new directory inside
            will be created for this experiment, and relevant data will be
            stored in there (experimental results, plots, etc).
        use_time: bool
            Whether to tack current date/time onto experiment directory name.
        params: dict (optional)
            A dictionary of parameters to be written to file, helping to
            identify the conditions that the experiment was run under.

    """
    def __init__(
            self, mode, base_estimators, x_var_name, x_var_values,
            generate_data, score=None, n_repeats=5, data_kwargs=None,
            search_kwargs=None, save_estimators=False, save_datasets=False,
            hook=None, name=None, directory=None, use_time=True, params=None):

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
            script_path = os.path.abspath(sys.modules['__main__'].__file__)
            name = os.path.splitext(os.path.basename(script_path))[0]
        self.name = name

        exp_store = ExperimentStore(directory or DEFAULT_EXPDIR)
        self.exp_dir = exp_store.new_experiment(self.name, use_time=use_time)

        self.save_estimators = save_estimators
        self.save_datasets = save_datasets
        self.hook = hook

        self.params = params

        try:
            self.script_path = os.path.abspath(sys.modules['__main__'].__file__)
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
        print("Running experiment called {}, "
              "storing in directory\n{}.".format(self.name, self.exp_dir.path))

        success = False

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
            success = True
        finally:
            self.exp_dir.finished(success)

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
            path = (
                'datasets/{x_var_name}={x_var_value}_round={round}'.format(
                    x_var_name=self.x_var_name,
                    x_var_value=x, round=r))
            self.exp_dir.save_object(path, data)

        if len(data) == 2:
            train, test = data
            context = None
        else:
            train, test, context = data

        return train, test, context

    def _train_and_test(
            self, base_est, x, r, train, test, context, search_rng, estimator_rng):

        est = clone(base_est)
        if self.mode == 'estimator':
            est.set_params(**{self.x_var_name: x})

        est_seed = sd.gen_seed(estimator_rng)
        est.random_state = est_seed

        est_path = None
        if 'directory' in est.get_params() or self.save_estimators:
            est_path = (
                'estimators/{est_name}_'
                '{x_var_name}={x_var_value}_'
                'round={round}'.format(
                    est_name=est.name, x_var_name=self.x_var_name,
                    x_var_value=x, round=r))
            full_est_path = self.exp_dir.path_for(est_path)

        if 'directory' in est.get_params():
            est.directory = full_est_path

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
            cv_score = 0.0
            cv_score_std = 0.0

            if n_iter == 1:
                est.set_params(**{k: v[0] for k, v in six.iteritems(dists)})
            learned_est = est

            then = time.time()
            learned_est.fit(train.X, train.y)
            logger.info("    Fit time: %s seconds." % (time.time() - then))
        else:
            if n_points > n_iter:
                search = RandomizedSearchCV(est, dists, scoring=self.scores[0], **self.search_kwargs)
                logger.info("    Running RandomizedSearchCV for {} iters...".format(n_iter))
            else:
                # These are arguments for RandomizedSearchCV but not GridSearchCV
                _search_kwargs = self.search_kwargs.copy()
                _search_kwargs.pop('random_state', None)
                _search_kwargs.pop('n_iter', None)

                search = GridSearchCV(est, dists, scoring=self.scores[0], **_search_kwargs)
                logger.info("    Running GridSearchCV for {} iters...".format(n_iter))

            then = time.time()
            search.fit(train.X, train.y)
            search_time_pp = (time.time() - then) / n_iter

            cv_score = search.cv_results_['mean_test_score'][search.best_index_]
            cv_score_std = search.cv_results_['std_test_score'][search.best_index_]

            learned_est = search.best_estimator_

            logger.info("    Best params: %s." % search.best_params_)
            logger.info("    Search time: %s seconds per point." % search_time_pp)

        if self.save_estimators:
            self.exp_dir.save_object(os.path.join(est_path, 'learned_est'), learned_est)

        results = {
            'round': r,
            self.x_var_name: x,
            'method': base_est.name,
            'search_time_per_point': search_time_pp}

        logger.info("    CV score mean: %f." % cv_score)
        results['cv_score'] = cv_score

        logger.info("    CV score std: %f." % cv_score_std)
        results['cv_score_std'] = cv_score_std

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
        x_var_display="", score_display=None, title="", labels=None, show=False, jitter_x=0.0):
    return __plot(
        experiment.score_names, experiment.x_var_name, df, plot_path, legend_loc,
        x_var_display, score_display, title, labels, show)


def __plot(
        score_names, x_var_name, df, plot_path, legend_loc='right',
        x_var_display="", score_display=None, title="", labels=None, show=False,
        jitter_x=0.0):

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
        x_var_display=x_var_display,
        jitter_x=float(jitter_x))
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
