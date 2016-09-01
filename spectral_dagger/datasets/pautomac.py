import os
import re
import sys
import numpy as np
from collections import defaultdict
import six
from itertools import product

from sklearn.utils import check_random_state

from spectral_dagger.sequence import ProbabilisticAutomaton
from spectral_dagger.sequence.pfa import is_pfa, is_dpfa
from spectral_dagger.sequence.hmm import is_hmm
from spectral_dagger.utils import rmse

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
        print "Is PFA?: ", is_pfa(model.b_0, model.B_o, model.b_inf_string)
        print "Is DPFA?: ", is_dpfa(model.b_0, model.B_o, model.b_inf_string)
        print "Is HMM?: ", is_hmm(model.b_0, model.B_o, model.b_inf_string)


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

    assert is_pfa(b_0, B_o, b_inf), "Loaded model is not a PFA."

    return ProbabilisticAutomaton(b_0, B_o, b_inf, estimator='string')


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


def _populate_prob_table(shape, density, alpha=1.0, random_state=None):
    """ Create a probability table.

    If len(shape)==2, we can think of the return value as a transition matrix.

    ``shape`` describes the shape of the returned probability table.
    Each ``row`` of the returned probability table (i.e. keeping the first
    n-1 dimensions fixed, moving along the n-th dimension) will be a discrete
    probability distribution. This distribution is selected by choosing
    ``density * shape[-1]`` of the values to be non-zero, and then selecting
    values for those non-zero location according to a Dirichlet distribution
    with parameter ``alpha``.

    Parameters
    ----------
    shape: list of integers
        Shape of probability table. We require len(shape) > 1.
    density: 0 < float < 1
        Percentage of non-zero entries.
    alpha: float > 0 or np vector > 0
        Parameters for dirichlet distribution.
        If a scalar, a symmetric dirichlet is used.
    random_state: np.random.RandomState
        Pseudo-random number generator.

    Returns
    -------
    probability table (ndarray)
    nonzero_indices (list)

    """
    assert len(shape) > 1
    random_state = check_random_state(random_state)

    n_nonzero_per_row = int(round(shape[-1] * density))
    try:
        len(alpha)
    except TypeError:
        alpha = alpha * np.ones(n_nonzero_per_row)

    row_indices = product(*[range(l) for l in shape[:-1]])

    probs = np.zeros(shape)
    nonzero_indices = []
    for idx in row_indices:
        row_values = random_state.dirichlet(alpha)
        row_indices = random_state.choice(
            shape[-1], n_nonzero_per_row, replace=False)
        probs[idx][row_indices] = row_values
        nonzero_indices.extend(idx + (i,) for i in row_indices)

    return probs, nonzero_indices


def make_pautomac_like(
        kind, n_states, n_symbols,
        symbol_density, transition_density,
        alpha=1.0, halts=False, random_state=None):
    """ Create a PFA using the same algo as the Pautomac competition.

    If ``halts`` is a number, then stopping probabilities are drawn from
    a ``beta(halts, 1)`` distribution. Otherwise, halting probabilities are
    determined by the same process as symbol probabilities.

    """
    assert kind in ['pfa', 'hmm'], "Cannot generate PFA of kind '%s'." % kind
    assert int(n_states) == n_states and n_states > 0
    assert int(n_symbols) == n_symbols and n_symbols > 0
    assert 0 < symbol_density < 1
    assert 0 < transition_density < 1
    assert alpha > 0

    random_state = check_random_state(random_state)

    symbols = range(n_symbols)

    n_start_states = int(round(transition_density * n_states))
    start_states = random_state.choice(n_states, n_start_states, replace=False)

    b_0 = np.zeros(n_states)
    b_0[start_states] = random_state.dirichlet(alpha * np.ones(n_start_states))

    if halts:
        if isinstance(halts, (float, int)):
            emission_probs, _ = _populate_prob_table(
                (n_states, n_symbols), symbol_density, alpha, random_state)
            halt_probs = random_state.beta(halts, 1, n_states)
            emission_probs *= (1 - halt_probs).reshape(-1, 1)
        else:
            emission_probs, _ = _populate_prob_table(
                (n_states, n_symbols+1), symbol_density, alpha, random_state)
            halt_probs = emission_probs[:, -1]
            emission_probs = emission_probs[:, :-1]
    else:
        emission_probs, _ = _populate_prob_table(
            (n_states, n_symbols), symbol_density, alpha, random_state)
        halt_probs = np.zeros(n_states)
    b_inf_string = halt_probs

    if kind == 'pfa':
        transition_probs, nonzero_indices = _populate_prob_table(
            (n_states, n_symbols, n_states), transition_density, alpha, random_state)
        B_o = {
            symbol: np.zeros((n_states, n_states)) for symbol in symbols}

        for state1, symbol, state2 in nonzero_indices:
            B_o[symbol][state1, state2] = (
                emission_probs[state1, symbol] *
                transition_probs[state1, symbol, state2])

        assert is_pfa(b_0, B_o, b_inf_string)

    elif kind == 'hmm':
        transition_probs, _ = _populate_prob_table(
            (n_states, n_states), transition_density, alpha, random_state)

        B_o = {
            s: np.diag(emission_probs[:, s]).dot(transition_probs)
            for s in symbols}

        assert is_hmm(b_0, B_o, b_inf_string)
    else:
        raise NotImplementedError(
            'Cannot generate PFA of kind "%s".' % kind)

    pa = ProbabilisticAutomaton(b_0, B_o, b_inf_string, estimator='string')
    return pa


if __name__ == "__main__":
    from spectral_dagger.sequence import top_k_basis, estimate_hankels
    from spectral_dagger.sequence.pfa import perturb_pfa_additive
    import pprint

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
