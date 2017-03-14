import logging
import os
import numpy as np
import pandas as pd
import argparse
import sys
import shutil
import traceback
from future.utils import raise_with_traceback
from sklearn.utils import check_random_state

from spectral_dagger.utils.experiment import Experiment, ExperimentStore, _plot, make_plot_hook
from spectral_dagger.utils.parallel import ParallelExperiment
from spectral_dagger.utils.misc import send_email_using_cfg

verbosity = 2
logger = logging.getLogger(__name__)
DEFAULT_EXPDIR = '/data/spectral_dagger/'


def _finish(experiment, email_cfg, success, **plot_kwargs):
    experiment.exp_dir.save_object('experiment', experiment)
    plot_path = experiment.exp_dir.path_for('plot.pdf')

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
        '--raise', action='store_true', dest='_raise',
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
        error_score='raise' if args._raise else -np.inf)

    if 'directory' not in exp_kwargs:
        exp_kwargs['directory'] = '/data/experiment'

    plot_kwargs['title'] = plot_kwargs.get('title', exp_kwargs['name'])
    exp_kwargs['hook'] = make_plot_hook('plot.pdf', **plot_kwargs)

    if args.parallel:
        experiment = ParallelExperiment(**exp_kwargs)
        experiment.build()

        archive_name = os.path.splitext(os.path.basename(experiment.exp_dir.path))[0]
        print("Zipping built experiment as {}.zip.".format(archive_name))
        shutil.make_archive(archive_name, 'zip', *os.path.split(experiment.exp_dir.path))

        print("Experiment has been built, execute using the ``sd-experiment`` command-line utility.")
        return

    if args.plot or args.plot_incomplete:
        # Re-plotting an experiment that has already been run.
        exp_store = ExperimentStore(exp_kwargs['directory'])
        kind = 'complete' if args.plot else 'incomplete'
        exp_dir = exp_store.get_latest_experiment(kind)

        plot_path = exp_dir.path_for('plot.pdf')
        df = pd.read_csv(exp_dir.path_for('results.csv'))
        experiment = exp_dir.load_object('experiment')

        fig = _plot(experiment, df, plot_path, **plot_kwargs)
    else:
        # Running a new experiment and then plotting.
        experiment = Experiment(**exp_kwargs)

        try:
            experiment.run('results.csv', random_state=random_state)
        except Exception as e:
            fig = _finish(experiment, args.email_cfg, False, **plot_kwargs)
            raise_with_traceback(e)

        fig = _finish(experiment, args.email_cfg, True, **plot_kwargs)

    return experiment, fig
