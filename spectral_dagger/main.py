import argparse
import os
import pandas as pd
import dill

from spectral_dagger.utils.experiment import ExperimentDirectory, _plot


def _plot_command(args):

    abs_path = os.path.abspath(args.path)
    directory, ed = os.path.split(abs_path)
    exp_dir = ExperimentDirectory(directory, ed)
    plot_path = exp_dir.path_for('plot.pdf')
    df = pd.read_csv(exp_dir.path_for('results.csv'))

    with open(exp_dir.path_for('experiment.pkl'), 'rb') as f:
        experiment = dill.load(f)

    plot_kwargs = dict(
        show=args.show, title=experiment.name,
        legend_loc='bottom')

    _plot(experiment, df, plot_path, **plot_kwargs)


def main():
    """ An entry point for using spectral dagger from the command line
        as ``sdagger`` to perform certain tasks related to spectral_dagger.

    """

    parser = argparse.ArgumentParser(
        'Execute commands related to the spectral_dagger package.')

    subparsers = parser.add_subparsers()

    plot_parser = subparsers.add_parser('plot', help='Plot an experiment.')
    plot_parser.add_argument('--path', type=str, default='.')
    plot_parser.add_argument('--show', action='store_true')
    plot_parser.set_defaults(func=_plot_command)

    args = parser.parse_args()
    args.func(args)
