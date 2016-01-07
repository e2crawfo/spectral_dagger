import utils
import numpy as np
import pandas as pd
from spectral_dagger.utils.math import normalize
from spectral_dagger.hmm import HMM, dummy_hmm

n_states = 4

results = []

O = np.eye(n_states)
O = normalize(O, ord=1, conservative=True)

T = np.random.random((n_states, n_states)) + np.eye(n_states)
T = normalize(T, ord=1, conservative=True)

n_states = T.shape[0]
n_obs = O.shape[1]
assert(T.shape[0] == T.shape[1] == O.shape[0])

observations = range(n_obs)
states = range(n_states)

init_dist = normalize(np.ones(n_states), ord=1, conservative=True)

hmm = HMM(observations, states, T, O, init_dist)

print "Rank of T: ", np.linalg.matrix_rank(hmm.T)
print "Rank of O: ", np.linalg.matrix_rank(hmm.O)
print "Number of states: ", hmm.T.shape[0]
print "Number of observations: ", hmm.O.shape[1]

n_training_samples = 100
n_testing_samples = 10
dimension_seq = range(1, 2*hmm.n_states)

# Doesn't actually make use of the testing_samples
score_func = utils.make_prediction_score(hmm, horizon=10)

basis_max_length = 2
horizon = 100
training_samples = [
    hmm.sample_trajectory(horizon) for i in range(n_training_samples)]
testing_samples = [
    hmm.sample_trajectory(horizon) for i in range(n_testing_samples)]

print "$" * 80
print "Basis max length: %d" % basis_max_length
print "$" * 80

# Do vanilla spectral learning
for e in ['string', 'prefix', 'substring']:
    model, n, score = utils.scan_dimensions(
        training_samples, testing_samples, hmm, dimension_seq,
        e, basis_max_length, score_func, utils.fit_true_hankels)

    print "^" * 40
    print "Estimated SVD with estimator %s" % e
    print "Dimension: ", n
    print "Horizon: ", horizon
    hankel_rank = np.linalg.matrix_rank(model.hankel.toarray(), tol=0.0001)
    print "Hankel Rank: ", hankel_rank
    print "Score: ", score
    print "^" * 40

    if e == 'substring':
        print model.hankel.toarray()

    results.append(dict(
        score=score, hankel_rank=hankel_rank, dimensions=n,
        horizon=horizon, basis_max_length=basis_max_length,
        estimator=e))

df = pd.DataFrame.from_records(results)
utils.all_hankels = {}
