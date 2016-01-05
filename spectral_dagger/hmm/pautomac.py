
import numpy as np

PAUTOMAC_PATH = (
    "/home/williamleif/Dropbox/icml2014-experiments/datasets/PAutomaC-competition_sets/")


def parse_pautomac_file(filename):
    fp = open(filename, "r")
    fp.readline()

    data = [[int(i) for i in line.split()[1:]] for line in fp]

    return data


def load_pautomac(problem):
    location = PAUTOMAC_PATH + str(problem)

    traindata = parse_pautomac_file(location + ".pautomac.train")
    testdata = parse_pautomac_file(location + ".pautomac.test")
    groundtruth = parse_groundtruth_file(location + ".pautomac_solution.txt")

    if metric == "KL":
    else:
        validdata = traindata[15000:20000]
        traindata = traindata[0:15000]


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

