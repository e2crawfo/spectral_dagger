import utils
import numpy as np

if __name__ == "__main__":
    print "Rank of T: ", np.linalg.matrix_rank(hmm.T)
    print "Rank of O: ", np.linalg.matrix_rank(hmm.O)
    print "Number of states: ", hmm.T.shape[0]
    print "Number of observations: ", hmm.O.shape[1]

    horizon = 3
    n_training_samples = 1000
    n_testing_samples = 10
    dimension_seq = range(20, hmm.n_states+1)

    training_samples = [
        hmm.sample_trajectory(horizon) for i in range(n_training_samples)]
    testing_samples = [
        hmm.sample_trajectory(horizon) for i in range(n_testing_samples)]

    # Doesn't actually make use of the testing_samples
    score_func = utils.make_prediction_score(hmm, horizon=3)

    for basis_max_length in range(1, 5):
        print "$" * 80
        print "Basis max length: %d" % basis_max_length
        print "$" * 80

        # Do vanilla spectral learning
        for e in ['string', 'prefix', 'substring']:
            model, n, score = utils.scan_dimensions(
                training_samples, testing_samples, hmm, dimension_seq,
                e, basis_max_length, score_func, utils.fit_estimated_svd)

            print "^" * 40
            print "Estimated SVD with estimator %s" % e
            print "Dimension: ", n
            print "Hankel Rank: ", (
                np.linalg.matrix_rank(model.hankel.toarray(), tol=0.0001))
            print "Score: ", score
            print "^" * 40
