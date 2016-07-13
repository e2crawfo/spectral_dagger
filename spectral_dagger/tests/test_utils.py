from spectral_dagger.utils.plot import (
    single_split_var, multiple_split_vars)
from spectral_dagger.utils.experiment import (
    data_experiment, estimator_experiment)


def test_plot_single():
    single_split_var()


def test_plot_multiple():
    multiple_split_vars()


def test_experiment_data():
    data_experiment()


def test_experiment_estimator():
    estimator_experiment()
