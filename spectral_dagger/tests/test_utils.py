import pytest
from sklearn.utils import check_random_state

from spectral_dagger import Estimator
from spectral_dagger.utils.experiment import Experiment, Dataset
from spectral_dagger.utils.plot import (
    single_split_var, multiple_split_vars)
from spectral_dagger.utils.misc import send_email
from spectral_dagger.examples.linear import (
    data_experiment, estimator_experiment)


def test_plot_single():
    single_split_var()


def test_plot_multiple():
    multiple_split_vars()


def test_experiment_data():
    data_experiment()


def test_experiment_estimator():
    estimator_experiment()


class DummyEstimator(Estimator):
    def __init__(self, a=10, arg=None, nested_est=None, name="Dummy", random_state=None):
        self._set_attrs(locals())

    def point_distribution(self, context=None):
        return {}

    def fit(self, X, y=None):
        self.arg_ = self.arg
        self.a_ = self.a if self.nested_est is None else self.nested_est.a
        return self

    def score(self, X, y=None):
        return self.a_


def _generate_data(arg=None, nested_arg=None, random_state=None):
    rng = check_random_state(random_state)
    if isinstance(nested_arg, dict):
        a = nested_arg['b']
        if isinstance(a, dict):
            a['p']

    return Dataset(rng.rand(10, 3)), Dataset(rng.rand(10, 3))


def test_experiment_hierarchical():
    e = Experiment(
        'estimator', [DummyEstimator(nested_est=DummyEstimator(a=20))], 'nested_est__a', [1, 2, 3],
        _generate_data, n_repeats=1, name="test_hierarchical:estimator",
        directory='/data/dummy')
    e.run()

    e = Experiment(
        'data', [DummyEstimator()], 'nested_arg__b__p', [1, 2, 3],
        _generate_data, n_repeats=1, name="test_hierarchical:data",
        directory='/data/dummy')
    e.run()


@pytest.mark.xfail
def test_email():
    host = "smtp.gmx.com"
    from_addr = "ml.experiment@gmx.com"
    password = ""
    to_addr = "wiricon@gmail.com"

    subject = "Test email from Python"
    body = "This is an email!"
    send_email(host, from_addr, password, subject, body, to_addr)
