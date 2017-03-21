import pprint
import logging
import six
import os
import time
import numpy as np
import pandas as pd
import dill as pickle
import sys
import glob

from sklearn.base import clone
from sklearn.utils import check_random_state
try:
    from sklearn.model_selection import (
        RandomizedSearchCV, GridSearchCV, ParameterGrid, ParameterSampler, cross_val_score)
except ImportError:
    from sklearn.grid_search import (
        RandomizedSearchCV, GridSearchCV, ParameterGrid, ParameterSampler, cross_val_score)

import spectral_dagger as sd
from spectral_dagger.utils.experiment import Experiment, get_n_points, __plot
from spectral_dagger.utils.misc import get_object_loader, ObjectSaver, str_int_list

logger = logging.getLogger(__name__)


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
        print("Building parallel experiment file for experiment {}, "
              "storing in directory:\n{}".format(self.name, self.exp_dir.path))
        self.saver = ObjectSaver(self.exp_dir.path, eager=True)

        rng = check_random_state(random_state)
        estimator_rng = check_random_state(sd.gen_seed(rng))
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

                    est_seed = sd.gen_seed(estimator_rng)
                    est.random_state = est_seed

                    train_indices = self._build_train_scenarios(
                        base_est, x, r, train_idx, context, search_rng)

                    self._build_test_scenario(
                        base_est, x, r, test_idx, train_indices, context)

        logging.getLogger('').removeHandler(handler)

        results = {
            'results': self.results, 'x_var_name': self.x_var_name, 'score_names': self.score_names}
        self.exp_dir.save_object('_results', results)

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
    scenario_type: str
        The type of scenario we are running.
    creates: str or list of str
        Key (or list thereof) that this scenario creates.

    """
    if isinstance(creates, str):
        creates = [creates]

    def f(w):
        def wrapper(input_archive, output_dir, scenario_idx, seed, verbose, redirect, force):
            if redirect:
                stdout_dir = os.path.join(output_dir, scenario_type+'_stdout')
                try:
                    os.makedirs(stdout_dir)
                except:
                    pass
                sys.stdout = open(os.path.join(stdout_dir, str(scenario_idx)), 'w')

                stderr_dir = os.path.join(output_dir, scenario_type+'_stderr')
                try:
                    os.makedirs(stderr_dir)
                except:
                    pass
                sys.stderr = open(os.path.join(stderr_dir, str(scenario_idx)), 'w')

            loader = get_object_loader(input_archive)
            finished = True
            for c in creates:
                try:
                    loader.load_object(c, scenario_idx)
                except KeyError:
                    finished = False
                    break

            if seed is not None:
                seed = scenario_idx + seed
            if finished and not force:
                print("Skipping {} scenario {} since it has "
                      "already been run.".format(scenario_type, scenario_idx))
            else:
                old_state = np.random.get_state()
                np.random.seed(seed)
                if verbose:
                    print("Beginning training scenario {}.".format(scenario_idx))

                w(input_archive, output_dir, scenario_idx)

                if verbose:
                    print("Done training scenario {}.".format(scenario_idx))
                np.random.set_state(old_state)

        return wrapper

    return f


def make_idx_print(idx):
    def idx_print(s):
        print("{}: {}".format(idx, s))
    return idx_print


@scenario('training', 'cv_score')
def run_training_scenario(input_archive, output_dir, scenario_idx):
    loader = get_object_loader(input_archive)

    estimator, kwargs = loader.load_object('train_scenario', scenario_idx)
    estimator.directory = os.path.join(output_dir, 'train_scratch/{}'.format(scenario_idx))
    dataset_idx = kwargs.pop('dataset_idx')
    train, train_kwargs = loader.load_object('train', dataset_idx)

    scoring = kwargs.get('scoring', None)
    cv = kwargs.get('cv', None)

    then = time.time()
    cv_score = cross_val_score(
        estimator, train.X, train.y, scoring=scoring, cv=cv, n_jobs=1, verbose=0)
    cv_time = time.time() - then
    make_idx_print(scenario_idx)("Cross-validation time: %s seconds." % cv_time)

    saver = ObjectSaver(output_dir, eager=True)
    saver.add_object(cv_score, 'cv_score', scenario_idx)
    saver.add_object(cv_time, 'cv_time', scenario_idx)


@scenario('testing', 'test_scores')
def run_testing_scenario(input_archive, output_dir, scenario_idx):
    # Choose winner of CVs, test each.
    loader = get_object_loader(input_archive)

    estimator, kwargs = loader.load_object('test_scenario', scenario_idx)
    estimator.directory = os.path.join(output_dir, 'test_scratch/{}'.format(scenario_idx))
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
    fit_time = time.time() - then
    idx_print("Fit time: %s seconds." % fit_time)

    test, test_kwargs = loader.load_object('test', dataset_idx)

    saver = ObjectSaver(output_dir, eager=True)

    idx_print("Running tests...")
    then = time.time()
    test_scores = {}
    for s, sn in zip(scores, score_names):
        score = s(estimator, test.X, test.y)
        idx_print("{},{},{}".format(scenario_idx, sn, score))
        test_scores[sn] = score

    test_time = time.time() - then
    idx_print("Test time: %s seconds." % test_time)

    test_scores['fit_time'] = fit_time
    test_scores['test_time'] = test_time

    saver.add_object(test_scores, 'test_scores', scenario_idx)


def parallel_exp_plot(directory, **plot_kwargs):
    results_file = os.path.join(directory, 'results.pkl')

    if os.path.exists(results_file):
        with open(os.path.join(directory, 'results.pkl'), 'r') as f:
            results = pickle.load(f)
    else:
        with open(os.path.join(directory, '_results.pkl'), 'r') as f:
            results = pickle.load(f)

        loader = get_object_loader(directory)
        test_scores = loader.load_objects_of_kind('test_scores')

        for test_scenario_idx, (ts, _) in list(test_scores.items()):
            for sn, s in list(ts.items()):
                results['results'][test_scenario_idx][sn] = s

        with open(os.path.join(directory, 'results.pkl'), 'w') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    score_names = results['score_names']
    x_var_name = results['x_var_name']

    df = pd.DataFrame.from_records(list(results['results'].values()))
    __plot(score_names, x_var_name, df, 'plot.pdf', **plot_kwargs)


def process_joblog(directory):
    joblog = os.path.join(directory, 'joblog.txt')
    df = pd.read_csv(joblog, sep='\t')
    print(df['JobRuntime'].describe())


def inspect_kind(directory, kind, idx):
    loader = get_object_loader(directory)

    if idx < 0:
        objects = loader.load_objects_of_kind(kind)
        objects = sorted([i for i in objects.items()], key=lambda x: x[0])
        print("Loaded object with kind {}.".format(kind, idx))

        for i, (o, k) in objects:
            print("Idx: {}".format(i))
            print("Object:")
            pprint.pprint(o)
            print("Associated kwargs:")
            pprint.pprint(k)
    else:
        obj, kwargs = loader.load_object(kind, idx)
        print("Loaded object with kind {} and idx {}.".format(kind, idx))
        if isinstance(obj, str):
            print("File contents:")
            print(obj)
        else:
            print("Object:")
            pprint.pprint(obj)
        print("Associated kwargs:")
        pprint.pprint(kwargs)


def sd_parallel(task, input_archive, output_dir, scenario_idx=-1, seed=None,
                force=0, verbose=0, redirect=1, **kwargs):

    logging.basicConfig(level=logging.INFO, format='')
    print("Running {} scenario {}.".format(task, scenario_idx))
    idx_print = make_idx_print(scenario_idx)
    idx_print("Listing directory.")
    idx_print(pprint.pformat(os.listdir('.')))
    idx_print("CWD: ")
    idx_print(pprint.pformat(os.getcwd()))

    if task == 'cv':
        run_training_scenario(input_archive, output_dir, scenario_idx, seed, verbose, redirect, force)
    elif task == 'test':
        run_testing_scenario(input_archive, output_dir, scenario_idx, seed, verbose, redirect, force)
    else:
        raise ValueError("Unknown task {} for sd-parallel.".format(task))


def sd_experiment(task, directory, scenario_idx=-1, **kwargs):

    logging.basicConfig(level=logging.INFO, format='')
    if task == 'plot':
        parallel_exp_plot(directory, **kwargs)
    elif task == 'joblog':
        process_joblog(directory)
    elif task == 'inspect':
        try:
            kind = kwargs['kind']
        except KeyError:
            raise KeyError("``kind`` not supplied, nothing to inspect.")
        try:
            idx = kwargs['idx']
        except KeyError:
            raise KeyError("``idx`` not supplied, nothing to inspect.")

        inspect_kind(directory, kind, idx)
    elif task == 'complete':
        directories = sorted(glob.glob(directory))
        for d in directories:
            print("\nChecking completion for:\n{}".format(d))

            loader = get_object_loader(d)

            train_scenario_idx = set(loader.indices_for_kind('train_scenario'))
            cv_score_idx = set(loader.indices_for_kind('cv_score'))
            unfinished = train_scenario_idx.difference(cv_score_idx)
            if unfinished:
                print("CV not finished. {} scenarios left to do.".format(len(unfinished)))
                print(str_int_list(unfinished))
            else:
                print("CV finished, {} scenarios complete.".format(len(train_scenario_idx)))

            test_scenario_idx = set(loader.indices_for_kind('test_scenario'))
            test_score_idx = set(loader.indices_for_kind('test_scores'))
            unfinished = test_scenario_idx.difference(test_score_idx)
            if unfinished:
                print("Testing not finished. {} scenarios left to do.".format(len(unfinished)))
                print(str_int_list(unfinished))
            else:
                print("Testing finished, {} scenarios complete.".format(len(test_scenario_idx)))
    else:
        raise ValueError("Unknown task {} for sd-experiment.".format(task))


def _sd_parallel():
    from clify import command_line
    command_line(sd_parallel, collect_kwargs=1, verbose=True)()


def _sd_experiment():
    from clify import command_line
    command_line(sd_experiment, collect_kwargs=1, verbose=True)()


if __name__ == "__main__":
    _sd_parallel()
