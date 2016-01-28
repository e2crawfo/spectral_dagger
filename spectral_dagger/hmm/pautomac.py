import os
import re
import sys
import numpy as np
from spectral_dagger.spectral import PredictiveStateRep

PAUTOMAC_PATH = "/data/PAutomaC-competition_sets/"


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
                start_states.append(map(float, vals[1:]))

            final_states = []
            for line in iter(lines.next, "S: (state,symbol)"):
                vals = re.split(split_pattern, line)
                final_states.append(map(float, vals[1:]))

            symbol_probs = []
            for line in iter(lines.next, "T: (state,symbol,state)"):
                vals = re.split(split_pattern, line)
                symbol_probs.append(map(float, vals[1:]))

            transition_probs = []
            for line in iter(f.readline, ""):
                vals = re.split(split_pattern, line)
                transition_probs.append(map(float, vals[1:]))

    except IOError:
        raise ValueError("No model exists with index %d." % problem_idx)

    n_obs = 1 + max([sym for state, sym, prob in symbol_probs])
    n_obs = int(n_obs)

    n_states = 1 + max([state for state, sym, prob in symbol_probs])
    n_states = int(n_states)

    obs_lookup = {(int(s), int(o)): prob for s, o, prob in symbol_probs}

    b_0 = np.zeros(n_states)
    for state, prob in start_states:
        b_0[int(state)] = prob

    b_inf = np.zeros(n_states)
    for state, prob in final_states:
        b_inf[int(state)] = prob

    B_o = {o: np.zeros((n_states, n_states)) for o in range(n_obs)}

    for s, o, s_prime, prob in transition_probs:
        s, o, s_prime = int(s), int(o), int(s_prime)

        obs_prob = obs_lookup.get((s, o), 0.0)
        halt_prob = b_inf[s]
        if halt_prob < 1.0:
            B_o[o][s, s_prime] = prob * obs_prob * (1 - halt_prob)

    # Check that the parsed WFA is a PFA.
    B = reduce(lambda x, y: x+y, B_o.values())
    B_row_sums = B.sum(axis=1)
    assert np.allclose(B_row_sums + b_inf, 1)

    return PredictiveStateRep(
        b_0, b_inf, B_o, can_terminate=sum(b_inf) > 0)


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


def load_pautomac(problem):
    location = PAUTOMAC_PATH + str(problem)

    traindata = load_pautomac_file(location + ".pautomac.train")
    testdata = load_pautomac_file(location + ".pautomac.test")
    groundtruth = parse_groundtruth_file(location + ".pautomac_solution.txt")

    if substring:
        basisdict = hankelmatrixcreator.top_k_basis(traindata,maxbasissize,n_symbols, 4)
        basislength = len(basisdict)
    else:
        prefixdict, suffixdict = hankelmatrixcreator.top_k_string_bases(traindata,maxbasissize,n_symbols)
        basislength = len(prefixdict)

    if substring:
        wfa = SparseSpectralWFA(n_symbols, basisdict, basisdict, 4)
    else:
        wfa = SparseSpectralWFA(n_symbols, prefixdict,suffixdict, 100)
    bestsize = 0
    avruntime = 0
    nummodelsmade = 0
    sizes = []
    begintime = time.clock()
    sizes.extend(range(10,31,10))
    for i in sizes:
        #if i == 5:
        if i == 10:
            wfa.fit(traindata, i,substring)
            inittime = wfa.inittime
        else:
            wfa.resize(traindata,i, substring)

        if metric == "WER":
            score = wfa.get_WER(validdata)
        else:
            score = wfa.scorepautomac(testdata,groundtruth)

        if bestsize == 0:
            bestscore = score
            bestsize = i
        elif score < bestscore:
            bestscore = score
            bestsize = i

        print "Model size: ", i, " Score: ", score
        avruntime += wfa.buildtime
        nummodelsmade += 1

    runtime = time.clock()-begintime

    if bestsize == 5:
        bestsize = 10;

    for i in range(int(bestsize)-9, int(bestsize)+10):

                    
        wfa.resize(traindata,i, substring)

        if metric == "WER":
            score = wfa.get_WER(testdata)
        else:
            score = wfa.scorepautomac(testdata,groundtruth)

        if bestsize == 0:
            bestwfa = copy.deepcopy(wfa)
            bestsize = i
        elif score < bestscore or math.isnan(bestscore):
            bestscore = score
            bestsize = i
        avruntime += wfa.buildtime
        nummodelsmade += 1

        print "Model size: ", i, " Score: ", score

    if metric == "WER":
        wfa.resize(traindata,i,substring)
        bestscore = wfa.get_WER(testdata)

    iohelpers.write_results(RESULTS_DIR+"spectral-"+esttype+"-pautomac="+problem+"-"+metric+".txt", problem, esttype+", "+"size= "+str(bestsize)+", basis size="+str(basislength), metric, bestscore, avruntime/float(nummodelsmade))

if __name__ == "__main__":
    from spectral_dagger.spectral.dynamical_system import PAStringGenerator
    from spectral_dagger.spectral import top_k_basis, estimate_hankels
    from spectral_dagger import sample_episodes, set_sim_rng
    import pprint

    set_sim_rng(np.random.RandomState(1))

    problem_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    print("Parsing problem %d." % problem_idx)
    pa = load_pautomac_model(problem_idx=problem_idx)
    generator = PAStringGenerator(pa)

    pp = pprint.PrettyPrinter()

    n_samples = 10000
    episodes = sample_episodes(n_samples, generator)

    basis = top_k_basis(episodes, 100, 'prefix')
    hankels = estimate_hankels(
        episodes, basis, generator.observations, 'prefix')
    hankel1 = hankels[2].toarray()

    # pautomac_train = load_pautomac_train(problem_idx)
    # hankels = estimate_hankels(
    #     pautomac_train, basis, generator.observations, 'prefix')
    # hankel2 = hankels[2].toarray()
