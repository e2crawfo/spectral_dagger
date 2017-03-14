import numpy as np
import os
from scipy.misc import logsumexp
from scipy.stats import multivariate_normal
import shutil
import warnings

from hmmlearn import hmm

from spectral_dagger import Space, Estimator
from spectral_dagger.sequence import SequenceModel
from spectral_dagger.utils.dists import MixtureDist
from spectral_dagger.datasets import pendigits

machine_eps = np.finfo(float).eps


class GaussianHMM(Estimator, hmm.GaussianHMM, SequenceModel):

    @property
    def record_attrs(self):
        return super(GaussianHMM, self).record_attrs or set(['n_components'])

    def point_distribution(self, context):
        pd = super(GaussianHMM, self).point_distribution(context)
        if 'max_states' in context:
            pd.update(n_components=list(range(2, context['max_states'])))
        return pd

    def fit(self, data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X = np.vstack(data)
            lengths = [len(d) for d in data]
            hmm.GaussianHMM.fit(self, X, lengths)

        self.obs_dists_ = [
            multivariate_normal(mean=m, cov=c)
            for m, c in zip(self.means_, self._get_covars())]

        return self

    @property
    def action_space(self):
        return None

    @property
    def observation_space(self):
        return Space([-np.inf, np.inf] * self.obs_dim, "ObsSpace")

    def has_reward(self):
        return False

    def has_terminal_states(self):
        return False

    def in_terminal_state(self):
        return False

    @property
    def obs_dim(self):
        return self.gmms[0].means_.shape[1]

    def check_terminal(self, o):
        return False

    def _reset(self, initial=None):
        self._b = np.log(self.startprob_)
        b = np.exp(self._b)
        assert np.isclose(b.sum(), 1), "Should sum to 1: %s, %f" % (b, b.sum())

        self._cond_obs_dist = None

    @property
    def log_T(self):
        try:
            self._log_T
        except AttributeError:
            self._log_T = np.log(self.transmat_)
        return self._log_T

    def update(self, o):
        old_b = self._b
        log_obs_prob = np.array([d.logpdf(o) for d in self.obs_dists_])
        log_obs_prob[np.isinf(log_obs_prob)] = -10000
        v = self._b.reshape(-1, 1) + self.log_T
        b = log_obs_prob + logsumexp(v, axis=0)
        self._b = b - logsumexp(b)
        b = np.exp(self._b)
        assert np.isclose(b.sum(), 1), (
            "Should sum to 1: %s, %f. "
            "Previous value: %s." % (b, b.sum(), np.exp(old_b)))
        self._cond_obs_dist = None

    def cond_obs_prob(self, o):
        """ Get probability of observing o given the current state. """
        return self.cond_obs_dist().pdf(o)

    def cond_termination_prob(self, o):
        return 0.0

    def cond_obs_dist(self):
        if self._cond_obs_dist is None:
            self._cond_obs_dist = MixtureDist(np.exp(self._b), self.obs_dists_)
        return self._cond_obs_dist

    def cond_predict(self):
        most_likely = np.argmax(self._b)
        return self.means_[most_likely]

    def predicts(self, seq):
        self.reset()
        predictions = []
        for o in seq:
            predictions.append(self.cond_predict())
            self.update(o)
        return np.array(predictions)

    def string_prob(self, string, log=False):
        log_prob = self.score(string)

        if log:
            return log_prob
        else:
            return np.exp(log_prob)

    def prefix_prob(self, string, log=False):
        return self.string_prob(string, log)


def qualitative(data, labels, model=None, n_repeats=1, dir_name=None, prefix=None):
    # Find an example of each digit, plot how the network does on each.
    unique_labels = list(set(labels))

    digits_to_plot = []
    for l in unique_labels:
        n_digits = 0
        for i in np.random.permutation(list(range(len(labels)))):
            if labels[i] == l:
                digits_to_plot.append((data[i], labels[i]))
                n_digits += 1
                if n_digits == n_repeats:
                    break

    i = 0
    for digit, label in digits_to_plot:
        plt.figure()
        title = "label=%d_round=%d" % (label, i)
        if prefix:
            title = prefix + '_' + title

        plt.subplot(2, 1, 1)
        pendigits.plot_digit(digit, difference=True)
        plt.title(title)
        plt.subplot(2, 1, 2)
        if model is not None:
            model.reset()
            prediction = [model.cond_predict()]
            for o in digit:
                model.update(o)
                prediction.append(model.cond_predict())
            plt.title("Prediction")
            pendigits.plot_digit(prediction, difference=True)
        if dir_name is None:
            plt.show()
        else:
            plt.savefig(os.path.join(dir_name, title+'.png'))
        plt.close()
        i += 1


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    use_digits = [0]
    difference = True
    simplify = True

    try:
        shutil.rmtree('hmm_images')
    except:
        pass

    dir_name = 'hmm_images'
    os.makedirs(dir_name)

    max_data_points = None

    data, labels = pendigits.get_data(difference=difference, use_digits=use_digits, simplify=5, sample_every=3)
    labels = [l for ll in labels[:10] for l in ll]
    data = [d for dd in data[:10] for d in dd]

    if max_data_points is None:
        max_data_points = len(data)
    perm = np.random.permutation(len(data))[:max_data_points]

    labels = [labels[i] for i in perm]
    data = [data[i] for i in perm]
    print(("Amount of data: ", len(data)))

    pct_test = 0.2
    n_train = int((1-pct_test) * len(data))

    test_data = data[n_train:]
    test_labels = labels[n_train:]

    data = data[:n_train]
    labels = labels[:n_train]

    n_components = 1

    for n_components in [10, 20, 30, 40, 50]:
        gmm_hmm = GaussianHMM(
            n_components=n_components,
            covariance_type='full', verbose=False, tol=1e-4, n_iter=1000)
        gmm_hmm.fit(data)
        print("Using %d states and digits: %s." % (n_components, use_digits))
        qualitative(
            data, labels, gmm_hmm, n_repeats=3,
            prefix='train_n_components=%d' % n_components, dir_name=dir_name)
        qualitative(
            test_data, test_labels, gmm_hmm, n_repeats=3,
            prefix='test_n_components=%d' % n_components, dir_name=dir_name)

        print(gmm_hmm.mean_log_likelihood(test_data))
        print(gmm_hmm.RMSE(test_data))

    # eps = gmm_hmm.sample_episodes(10, horizon=int(np.mean([len(s) for s in data])))
    # for s in eps:
    #     if s:
    #         pendigits.plot_digit(s, difference)
    #         plt.show()
