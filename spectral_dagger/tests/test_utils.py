import pytest

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


@pytest.mark.xfail
def test_email():
    host = "smtp.gmx.com"
    from_addr = "ml.experiment@gmx.com"
    password = ""
    to_addr = "wiricon@gmail.com"

    subject = "Test email from Python"
    body = "This is an email!"
    send_email(host, from_addr, password, subject, body, to_addr)
