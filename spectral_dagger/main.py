import pandas as pd

from spectral_dagger.utils.experiment import ExperimentDirectory, _plot


def _plot_command(path, show):
    exp_dir = ExperimentDirectory(path)
    plot_path = exp_dir.path_for('plot.pdf')
    df = pd.read_csv(exp_dir.path_for('results.csv'))
    experiment = exp_dir.load_object('experiment')

    _plot(
        experiment, df, plot_path,
        show=show, title=experiment.name, legend_loc='bottom')


def main():
    """ An entry point for using spectral dagger from the command line
        as ``sdagger`` to perform certain tasks related to spectral_dagger.

    """
    from clify import command_line
    command_line(_plot_command)()
