from __future__ import print_function
import numpy as np
import pprint
import logging
import sklearn
from sklearn.cross_validation import cross_val_score, KFold
import abc
import six

pp = pprint.PrettyPrinter()
verbosity = 2

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def update_attrs(obj, attrs):
    for k, v in six.iteritems(attrs):
        setattr(obj, k, v)


class Estimator(object):
    __metaclass__ = abc.ABCMeta

    record_attrs = []

    @abc.abstractmethod
    def _generate_points(self, max_evals, context, rng):
        raise NotImplementedError()

    def generate_points(self, max_evals, context, rng):
        points = self._generate_points(max_evals, context, rng)
        points = [pt for pt in points if pt]

        return points if len(points) <= max_evals else points[:max_evals]


class Dataset(object):
    """ A generic dataset.

    Parameters/Attributes
    ---------------------
    X: any
        Input data.
    y: any
        Output data.

    """
    def __init__(self, X, y):
        if len(X) == 1:
            X = [X]
        self.X = X

        if len(y) == 1:
            y = [y]
        self.y = y

        assert len(self.X) == len(self.y)

    def __len__(self):
        return len(self.X)

    def __str__(self):
        return "<Dataset. len=%d>" % len(self)

    def __repr__(self):
        return str(self)

    def __iter__(self):
        return [(x, y) for x, y in zip(self.X, self.y)]

    def __getitem__(self, key):
        return Dataset(self.X[key], self.y[key])


class UnsupervisedDataset(Dataset):
    """ A generic dataset.

    Parameters/Attributes
    ---------------------
    X: any
        Data.

    """
    def __init__(self, X):
        if len(X) == 1:
            X = [X]
        self.X = X
        self.y = None

    def __iter__(self):
        return [x for x in self.X]

    def __getitem__(self, key):
        return UnsupervisedDataset(self.X[key])


class ModelSelector(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    def add_training_data(self, train, context=None):
        """ Add training data to the ModelSelector.

        Calling this fn replaces any previously added data.

        ``context`` is a dictionary containing additional information about
        how the data was generated which may be relevant for choosing
        ranges of hyperparameters to experiment with.

        """
        self.train = train
        self.context = context

    @abc.abstractmethod
    def select(self, base_est):
        raise NotImplementedError()


class CrossValidator(ModelSelector):
    """ An instance of this class specifies cross validation settings,
    and can be used repeatedly with different estimators (to easily test
    multiple estimators under the same cross validation settings).

    Parameters
    ----------
    cv: function
        A function which accepts a number of training points and Returns
        a cross validation iterator. Signature: ``cv(n_train)``.

    """
    def __init__(
            self, cv=None, n_jobs=1, max_evals=np.inf,
            use_keys=False, threshold=None, rng=None):

        self.cv = cv
        self.n_jobs = n_jobs

        self.max_evals = max_evals
        self.threshold = np.inf if threshold is None else threshold

        self.use_keys = use_keys
        self.rng = rng

    def select(self, base_est, score=None):
        """ Compute cross-validation loss for estimator ``base_est``
        at different parameter settings.

        Parameter settings are created by calling base_est.generate_points().
        A maximum of ``self.max_evals`` points are tested.

        Parameters
        ----------
        base_est: Estimator instance
            The estimator to test. Will be cloned for each hyperparameter
            setting that we test during cross validation.
        score: function
            A function which returns the score of a set of predictions on a
            Dataset. Signature: TODO

        Returns
        -------
        A dict.

        """
        results = []

        points = base_est.generate_points(
            self.max_evals, context=self.context, rng=self.rng)

        try:
            for point in points:
                est = sklearn.base.clone(base_est)
                update_attrs(est, point)

                cv_scores = cross_val_score(
                    est, X=self.train.X, y=self.train.y,
                    scoring=score, cv=self.cv(len(self.train.X)),
                    n_jobs=self.n_jobs)

                r = dict(
                    score=np.mean(cv_scores),
                    scores=cv_scores,
                    estimator=est)

                results.append(r)

        except StopIteration:
            pass

        best = max(results, key=lambda d: d['score'])

        # Store results so they can be accessed later
        self.results = results
        self.best = best

        model = sklearn.base.clone(best['estimator'])
        model.fit(self.train.X, self.train.y)

        return model


if __name__ == "__main__":
    """ Implement the sklearn LASSO examples using our cross validator. """

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import datasets, linear_model

    diabetes = datasets.load_diabetes()
    X = diabetes.data[:150]
    y = diabetes.target[:150]

    class Lasso(linear_model.Lasso, Estimator):
        record_attrs = ['alpha']

        def _generate_points(self, max_evals, context, rng):
            return [dict(alpha=a) for a in np.logspace(-4, -.5, 30)]

    scores = list()
    scores_std = list()

    def f(n_points):
        return KFold(n_points, n_folds=3, shuffle=False)
    selector = CrossValidator(f)
    selector.add_training_data(Dataset(X, y))
    model = selector.select(Lasso(0.0))

    # alpha made directly available through ``record_attrs``.
    alphas = [r['estimator'].alpha for r in selector.results]
    scores = [r['score'] for r in selector.results]
    scores_std = [np.std(r['scores']) for r in selector.results]

    plt.figure(figsize=(4, 3))
    plt.semilogx(alphas, scores)
    # plot error lines showing +/- std. errors of the scores
    plt.semilogx(alphas, np.array(scores) + np.array(scores_std) / np.sqrt(len(X)),
                 'b--')
    plt.semilogx(alphas, np.array(scores) - np.array(scores_std) / np.sqrt(len(X)),
                 'b--')
    plt.ylabel('CV score')
    plt.xlabel('alpha')
    plt.axhline(np.max(scores), linestyle='--', color='.5')
    plt.show()
