import os
import re
import sys
import numpy as np
from collections import defaultdict
import six
from itertools import product
import operator

from spectral_dagger.sequence import ProbabilisticAutomaton
from spectral_dagger.sequence.pfa import is_pfa, is_dpfa
from spectral_dagger.sequence.hmm import is_hmm
from spectral_dagger.utils import rmse, default_rng

PAUTOMAC_PATH = "/data/PAutomaC-competition_sets/"


def pautomac_available():
    return bool(problem_indices())


def problem_indices():
    fnames = sorted(os.listdir(PAUTOMAC_PATH))

    if fnames[0].startswith('00'):
        fnames = fnames[1:]

    problem_files = defaultdict(list)
    for f in fnames:
        problem_idx = int(f[:f.find('.')])
        problem_files[problem_idx].append(f)

    problem_indices = []
    for idx, files in six.iteritems(problem_files):
        if len(files) == 4:
            problem_indices.append(idx)

    return sorted(problem_indices)


def all_models():
    return [load_pautomac_model(idx) for idx in problem_indices()]


def print_models():
    for i, model in enumerate(all_models()):
        print str(i+1) + ": " + "*" * 40
        print model
        print "Is PFA?: ", is_pfa(model.b_0, model.b_inf_string, model.B_o)
        print "Is DPFA?: ", is_dpfa(model.b_0, model.b_inf_string, model.B_o)
        print "Is HMM?: ", is_hmm(model.b_0, model.b_inf_string, model.B_o)


def int_or_float(x):
    try:
        return int(x)
    except ValueError:
        return float(x)


def load_pautomac_model(problem_idx):
    """ Returns a Probabilistic Automaton. """

    try:
        split_pattern = "[(), ]+"

        fname = os.path.join(
            PAUTOMAC_PATH, "%d.pautomac_model.txt" % problem_idx)

        with open(fname, 'r') as f:
            lines = (line.strip() for line in iter(f.readline, ""))
            lines.next()

            start_states = []
            for line in iter(lines.next, "F: (state)"):
                vals = re.split(split_pattern, line)
                start_states.append(map(int_or_float, vals[1:]))

            # Probability of emitting symbol, given that
            # we are in state and not halting
            final_states = []
            for line in iter(lines.next, "S: (state,symbol)"):
                vals = re.split(split_pattern, line)
                final_states.append(map(int_or_float, vals[1:]))

            # Probability of transitioning to state2, given that we are in
            # state1 and emitting symbol and not halting
            symbol_probs = []
            for line in iter(lines.next, "T: (state,symbol,state)"):
                vals = re.split(split_pattern, line)
                symbol_probs.append(map(int_or_float, vals[1:]))

            transition_probs = []
            for line in iter(f.readline, ""):
                vals = re.split(split_pattern, line)
                transition_probs.append(map(int_or_float, vals[1:]))

    except IOError:
        raise ValueError("No model exists with index %d." % problem_idx)

    n_obs = 1 + max([sym for state, sym, prob in symbol_probs])

    n_states = 1 + max([state for state, sym, prob in symbol_probs])
    n_states = n_states

    obs_lookup = {(s, o): prob for s, o, prob in symbol_probs}

    b_0 = np.zeros(n_states)
    for state, prob in start_states:
        b_0[state] = prob

    b_inf = np.zeros(n_states)
    for state, prob in final_states:
        b_inf[state] = prob

    B_o = {o: np.zeros((n_states, n_states)) for o in range(n_obs)}

    for s, o, s_prime, prob in transition_probs:
        obs_prob = obs_lookup.get((s, o), 0.0)
        halt_prob = b_inf[s]
        if halt_prob < 1.0:
            B_o[o][s, s_prime] = prob * obs_prob * (1 - halt_prob)

    assert is_pfa(b_0, b_inf, B_o), "Loaded model is not a PFA."

    return ProbabilisticAutomaton(b_0, b_inf, B_o, estimator='string')


def load_pautomac_ground_truth(problem_idx):
    fname = os.path.join(
        PAUTOMAC_PATH, "%d.pautomac_solution.txt" % problem_idx)
    fp = open(fname, "r")
    fp.readline()
    true_probs = [float(line) for line in fp]
    return true_probs


def load_pautomac_train(problem_idx):
    fname = os.path.join(
        PAUTOMAC_PATH, "%d.pautomac.train" % problem_idx)
    return load_pautomac_file(fname)


def load_pautomac_test(problem_idx):
    fname = os.path.join(
        PAUTOMAC_PATH, "%d.pautomac.test" % problem_idx)
    return load_pautomac_file(fname)


def load_pautomac_file(filename):
    fp = open(filename, "r")
    fp.readline()

    data = [[int(i) for i in line.split()[1:]] for line in fp]

    return data


def pautomac_score(model, problem_idx):
    """ Score a model using PAutomaC ground truth. """

    test_data = load_pautomac_test(problem_idx)
    ground_truth = load_pautomac_ground_truth(problem_idx)

    model_probs = np.array([
        model.get_string_prob(seq) for seq in test_data])

    # The PAutomaC paper (Verwer et al. 2012) requests that the
    # submitted probabilities be normalized to sum to 1. The ground
    # truth probabilities are also normalized in this way.
    model_probs /= model_probs.sum()

    expected_llh = 0.0
    for gt, mp in zip(ground_truth, model_probs):
        expected_llh += gt * np.log2(mp)

    return 2**(-expected_llh)


def _populate_prob_table(lengths, sparsity, rng):
    """ Create a probability table with dimensions given by `lengths`.

    For every possible combination of the first n-1 indices, where `lengths`
    has length n, the probability table contains a row that is a normalized
    probability distribution. The nonzero entries in this row are chosen
    randomly before the table is created according to the sparsity.
    The values for the non-zero values are chosen from a Dirichlet
    distribution. If some combination of the first n-1 indices has no
    nonzero values, we create one for it artificially.

    Parameters
    ----------
    lengths: list of integers
        Size of each dimension.
    sparsity: 0 < float < 1
        Percentage of non-zero entries.
    rng: np.random.RandomState
        Pseudo-random number generator.

    """
    lengths_prod = reduce(operator.mul, lengths, 1)
    n_nonzero = int(round(sparsity * lengths_prod))

    nonzero_set = set(rng.choice(
        lengths_prod, n_nonzero, replace=False))
    nonzero_indices = [
        p for i, p in enumerate(product(*[range(l) for l in lengths]))
        if i in nonzero_set]

    nonzero_table = {
        t: [] for t in product(*[range(l) for l in lengths[:-1]])}
    for t in nonzero_indices:
        nonzero_table[t[:-1]].append(t[-1])

    probs = np.zeros(lengths)
    for t_trunc, indices in six.iteritems(nonzero_table):
        if not indices:
            indices = [rng.choice(lengths[-1])]
            nonzero_indices.append(t_trunc + (indices[0],))

        row = rng.dirichlet([1] * len(indices))
        for i, idx in enumerate(indices):
            probs[t_trunc + (idx,)] = row[i]

    return probs, nonzero_indices


def make_pautomac_like(
        kind, n_states, n_symbols,
        symbol_sparsity, transition_sparsity, halts=False, rng=None):
    """ Create a PFA using the same algo as the Pautomac competition.

    N.B: doesn't correctly handle the case where halts=True. """

    assert int(n_states) == n_states and n_states > 0
    assert int(n_symbols) == n_symbols and n_symbols > 0
    assert 0 < symbol_sparsity < 1
    assert 0 < transition_sparsity < 1

    rng = default_rng(rng)

    symbols = range(n_symbols)

    n_start_states = int(round(transition_sparsity * n_states))
    start_states = rng.choice(n_states, n_start_states, replace=False)

    b_0 = np.zeros(n_states)
    b_0[start_states] = rng.dirichlet([1] * n_start_states)

    if halts:
        n_halt_states = int(round(transition_sparsity * n_states))
        halt_states = rng.choice(n_states, n_halt_states, replace=False)
    else:
        n_halt_states = 0
        halt_states = []

    b_inf_string = np.zeros(n_states)
    b_inf_string[halt_states] = rng.dirichlet([1] * n_halt_states)

    emission_probs, _ = _populate_prob_table(
        (n_states, n_symbols), symbol_sparsity, rng)

    if kind == 'pfa':
        transition_probs, nonzero_indices = _populate_prob_table(
            (n_states, n_symbols, n_states), transition_sparsity, rng)
        B_o = {
            symbol: np.zeros((n_states, n_states)) for symbol in symbols}

        for state1, symbol, state2 in nonzero_indices:
            B_o[symbol][state1, state2] = (
                emission_probs[state1, symbol] *
                transition_probs[state1, symbol, state2])

        assert is_pfa(b_0, b_inf_string, B_o)

    elif kind == 'hmm':
        transition_probs, _ = _populate_prob_table(
            (n_states, n_states), transition_sparsity, rng)

        B_o = {
            s: np.diag(emission_probs[:, s]).dot(transition_probs)
            for s in symbols}

        assert is_hmm(b_0, b_inf_string, B_o)
    else:
        raise NotImplementedError(
            'Cannot generate PFA of kind "%s".' % kind)

    pa = ProbabilisticAutomaton(b_0, b_inf_string, B_o, estimator='string')
    return pa


if __name__ == "__main__":
    from spectral_dagger.sequence import top_k_basis, estimate_hankels
    from spectral_dagger.sequence.pfa import perturb_pfa_additive
    from spectral_dagger import set_sim_rng
    import pprint

    set_sim_rng(np.random.RandomState(1))

    problem_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    print("Parsing problem %d." % problem_idx)
    pa = load_pautomac_model(problem_idx=problem_idx)
    pp = pprint.PrettyPrinter()

    print("Generated from model. " + "=" * 40)
    n_samples = 10000
    episodes = pa.sample_episodes(n_samples)

    basis = top_k_basis(episodes, 100, 'prefix')
    hankels = estimate_hankels(
        episodes, basis, pa.observations, 'prefix')
    hankel1 = hankels[2].toarray()
    pp.pprint(hankel1)

    print("True samples. " + "=" * 40)
    pautomac_train = load_pautomac_train(problem_idx)
    hankels = estimate_hankels(
        pautomac_train, basis, pa.observations, 'prefix')
    hankel2 = hankels[2].toarray()
    pp.pprint(hankel2)

    print("Perturbed model. " + "=" * 40)
    pert = perturb_pfa_additive(pa, noise=1.0/pa.b_0.size**0.01)

    n_samples = 10000
    episodes = pert.sample_episodes(n_samples)

    hankels = estimate_hankels(
        episodes, basis, pa.observations, 'prefix')
    hankel3 = hankels[2].toarray()
    pp.pprint(hankel3)

    print("Generated from model again. " + "=" * 40)
    n_samples = 10000
    episodes = pa.sample_episodes(n_samples)

    basis = top_k_basis(episodes, 100, 'prefix')
    hankels = estimate_hankels(
        episodes, basis, pa.observations, 'prefix')
    hankel4 = hankels[2].toarray()
    pp.pprint(hankel4)

    print("RMSE estimated Hankel: ", rmse(hankel1, hankel2))
    print("RMSE perturbed Hankel: ", rmse(hankel1, hankel3))
    print("RMSE re-estimated Hankel: ", rmse(hankel2, hankel4))
