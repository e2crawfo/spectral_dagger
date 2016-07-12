from __future__ import print_function
import numpy as np
import pandas as pd
import pprint
import logging
import abc
import six
import sklearn
from sklearn.cross_validation import train_test_split, KFold
from sklearn.metrics import make_scorer

import spectral_dagger as sd
from spectral_dagger.utils.cv import Dataset, CrossValidator, Estimator

pp = pprint.PrettyPrinter()
verbosity = 2

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Experiment(object):

    @abc.abstractmethod
    def run(self):
        raise NotImplementedError()


def default_score(estimator, X, y):
    return estimator.score(X, y)


class DataExperiment(object):
    """ Run an experiment which explores how the performance of one or more
        base_estimators changes as some parameter of the data generation
        process is varied.

        Parameters
        ----------
        base_estimators: Estimator class or list of Estimator classes.
            The base_estimators to be tested.
        model_selector: ModelSelector instance
            An object for selecting hypotheses.
        x_var_name: string
            Name to use for the x variable (aka independent variable).
        x_var_values: list-like
            Values that the x variable can take on.
        generate_data: function
            A function which generates data. Must accept an argument with
            name equal to ``x_var_name``.
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
        seed: int or RandomState
            Random state for the experiment.
        data_kwargs: dict
            Key word arguments for the data generation function.

    """
    def __init__(
            self, base_estimators, model_selector, x_var_name, x_var_values,
            generate_data, score, n_repeats=5, seed=1, data_kwargs=None):

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
                    self.scores.append(s[0])
                    self.score_names.append(s[1])
                else:
                    self.scores.append(s)
                    self.score_names.append(
                        s.__name__ if hasattr(s, "__name__") else str(s))
        else:
            self.scores = [default_score]
            self.score_names = ['score']

        self.score = score[0]

        self.n_repeats = n_repeats
        self.model_selector = model_selector

        self.seed = seed

        self.data_kwargs = {} if data_kwargs is None else data_kwargs

    def run(self):
        print("Running experiment.")
        print("Args: ")
        print(locals())

        rng = np.random.RandomState(self.seed)

        results = []
        for x in self.x_var_values:
            for i in range(self.n_repeats):
                data_seed = sd.gen_seed(rng)
                data_kwargs = self.data_kwargs.copy()
                data_kwargs.update(
                    {self.x_var_name: x, 'seed': data_seed})

                train, test = self.generate_data(**data_kwargs)
                self.model_selector.add_training_data(train)

                for base_est in self.base_estimators:
                    est = sklearn.base.clone(base_est)
                    est.random_state = sd.gen_seed(rng)

                    print(
                        "Collecting data point. "
                        "method: %s, %s: %s, repeat: %d, seed: %d."
                        "" % (base_est.__class__.__name__, self.x_var_name,
                              str(x), i, est.random_state))

                    learned_est = (
                        self.model_selector.select(est, self.scores[0]))

                    print("    Best parameter setting:"
                          " %s" % learned_est.get_params())

                    results.append({
                        'round': i, self.x_var_name: x,
                        'method': base_est.__class__.__name__})

                    for s, sn in zip(self.scores, self.score_names):
                        score = s(learned_est, test.X, test.y)
                        print("    Test score %s: %f" % (sn, score))
                        results[-1][sn] = score

                    for attr in learned_est.record_attrs:
                        results[-1][attr] = getattr(learned_est, attr)

        self.results = results
        self.df = pd.DataFrame.from_records(results)

        return self.df


if __name__ == "__main__":
    """ Explore how lasso and ridge regression perform as the
    amount of data available to them changes. """

    import matplotlib.pyplot as plt
    from sklearn import datasets, linear_model
    from plot import plot_measures

    def generate_diabetes_data(train_size, seed):
        diabetes = datasets.load_diabetes()
        X_train, X_test, y_train, y_test = train_test_split(
            diabetes.data, diabetes.target,
            train_size=train_size, test_size=0.2, random_state=seed)

        return Dataset(X_train, y_train), Dataset(X_test, y_test)

    mse_score = make_scorer(
        sklearn.metrics.mean_squared_error, greater_is_better=False)
    mae_score = make_scorer(
        sklearn.metrics.mean_absolute_error, greater_is_better=False)

    class Ridge(linear_model.Ridge, Estimator):
        record_attrs = ['alpha']

        def _generate_points(self, max_evals, context, rng):
            return [dict(alpha=a) for a in np.logspace(-4, -.5, 30)]

    class Lasso(linear_model.Lasso, Estimator):
        record_attrs = ['alpha']

        def _generate_points(self, max_evals, context, rng):
            return [dict(alpha=a) for a in np.logspace(-4, -.5, 30)]

    def f(n_points):
        return KFold(n_points, n_folds=10, shuffle=False)

    selector = CrossValidator(f, n_jobs=1)

    experiment = DataExperiment(
        [Ridge(0.1), Lasso(0.1)],
        selector, 'train_size', np.linspace(0.1, 0.8, 8),
        generate_diabetes_data, [(mse_score, 'NMSE'), (mae_score, 'NMAE')])
    df = experiment.run()

    plot_measures(df, ['NMSE', 'NMAE', 'alpha'], 'train_size', 'method')
    plt.show()
