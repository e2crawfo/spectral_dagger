import os
import re
import sys
import numpy as np
from collections import defaultdict
import six

from spectral_dagger.sequence import PredictiveStateRep
from spectral_dagger.sequence.pfa import is_pfa, is_dpfa, is_hmm
from spectral_dagger.utils import rmse

PAUTOMAC_PATH = "/data/PAutomaC-competition_sets/"


def pautomac_available():
    return bool(problem_indices)


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

            final_states = []
            for line in iter(lines.next, "S: (state,symbol)"):
                vals = re.split(split_pattern, line)
                final_states.append(map(int_or_float, vals[1:]))

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

    return PredictiveStateRep(b_0, b_inf, B_o, estimator='string')


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


if __name__ == "__main__":
    from spectral_dagger.sequence.pfa import PFASampler
    from spectral_dagger.sequence import top_k_basis, estimate_hankels
    from spectral_dagger.sequence.pfa import perturb_pfa_additive
    from spectral_dagger import set_sim_rng
    import pprint

    set_sim_rng(np.random.RandomState(1))

    problem_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    print("Parsing problem %d." % problem_idx)
    pa = load_pautomac_model(problem_idx=problem_idx)
    generator = PFASampler(pa)

    pp = pprint.PrettyPrinter()

    print("Generated from model. " + "=" * 40)
    n_samples = 10000
    episodes = generator.sample_episodes(n_samples)

    basis = top_k_basis(episodes, 100, 'prefix')
    hankels = estimate_hankels(
        episodes, basis, generator.observations, 'prefix')
    hankel1 = hankels[2].toarray()
    pp.pprint(hankel1)

    print("True samples. " + "=" * 40)
    pautomac_train = load_pautomac_train(problem_idx)
    hankels = estimate_hankels(
        pautomac_train, basis, generator.observations, 'prefix')
    hankel2 = hankels[2].toarray()
    pp.pprint(hankel2)

    print("Perturbed model. " + "=" * 40)
    pert = perturb_pfa_additive(pa, noise=1.0/pa.b_0.size**0.01)
    pert_gen = PFASampler(pert)

    n_samples = 10000
    episodes = pert_gen.sample_episodes(n_samples)

    hankels = estimate_hankels(
        episodes, basis, generator.observations, 'prefix')
    hankel3 = hankels[2].toarray()
    pp.pprint(hankel3)

    print("Generated from model again. " + "=" * 40)
    n_samples = 10000
    episodes = generator.sample_episodes(n_samples)

    basis = top_k_basis(episodes, 100, 'prefix')
    hankels = estimate_hankels(
        episodes, basis, generator.observations, 'prefix')
    hankel4 = hankels[2].toarray()
    pp.pprint(hankel4)

    print("RMSE estimated Hankel: ", rmse(hankel1, hankel2))
    print("RMSE perturbed Hankel: ", rmse(hankel1, hankel3))
    print("RMSE re-estimated Hankel: ", rmse(hankel2, hankel4))
