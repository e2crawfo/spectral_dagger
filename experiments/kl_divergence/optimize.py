from hyperopt import fmin, tpe, hp
from hyperopt.mongoexp import MongoTrials
from hyperopt import STATUS_OK

import time
import subprocess
import argparse

from kl_divergence import f as kl_divergence


if __name__ == "__main__":
    description = 'Optimize parameters for kl_divergence.'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--mongo-workers',
                        dest='mongo_workers',
                        default=8,
                        type=int,
                        help='Number of parallel workers to use to evaluate '
                             'points. Only has an effect if --mongo is also '
                             'supplied')

    parser.add_argument('--exp-key',
                        dest='exp_key',
                        default='exp1',
                        type=str,
                        help='Unique key identifying this experiment within '
                             'the mongodb')

    parser.add_argument('--dry-run',
                        dest='dry_run',
                        action='store_true',
                        help='A dry run will not evaluate the function. '
                             'Useful for testing the hyperopt framework '
                             'without having to wait for the function '
                             'evaluation')

    argvals = parser.parse_args()

    num_mongo_workers = max(argvals.mongo_workers, 1)

    exp_key = argvals.exp_key
    dry_run = argvals.dry_run

    N = 20
    num_samples = 20

    if dry_run:
        def make_f():
            from hyperopt import STATUS_OK

            def f(x):
                return {'loss': 0, 'loss_variance': 0, 'status': STATUS_OK}

            return f

        objective_func = make_f()

    else:
        objective_func = kl_divergence

    trials = MongoTrials('mongo://localhost:1234/assoc/jobs', exp_key=exp_key)
    print "Trials: " + str(trials.trials) + "\n"
    print "Results: " + str(trials.results) + "\n"
    print "Losses: " + str(trials.losses()) + "\n"
    print "Statuses: " + str(trials.statuses()) + "\n"

    if trials:
        key_func = lambda x: x[u'result'].get(u'loss', 1000)
        best = min(trials.trials, key=key_func)

        print "Best: "
        print "Result: ", best[u'result']
        print "Vals: ", best[u'misc'][u'vals']

    worker_call_string = ["hyperopt-mongo-worker",
                          "--mongo=localhost:1234/assoc",
                          "--max-consecutive-failures", "2",
                          "--reserve-timeout", "2.0",
                          "--workdir","~/spectral_dagger/experiments/kl_divergence/",
                          ]

    print "Worker Call String"
    print worker_call_string
    workers = []
    for i in range(num_mongo_workers):
        p = subprocess.Popen(worker_call_string)
        workers.append(p)

    space = hp.quniform('n_extra_actions', 0, 10)

    then = time.time()

    print "Calling fMin"
    best = fmin(objective_func,
                space=space,
                algo=tpe.suggest,
                max_evals=1000,
                trials=trials)
    print "Done fMin"

    now = time.time()

    #directory = '/home/e2crawfo/Dropbox/projects/cleanup_parameters/results'
    #filename = fh.make_filename('optlog', directory=directory, use_time=True)
    #aggregated_log = open(filename, 'w')

    #aggregated_log.write("Time for fmin: " + str(now - then) + "\n")
    #aggregated_log.write("Trials: " + str(trials.trials) + "\n")
    #aggregated_log.write("Results: " + str(trials.results) + "\n")
    #aggregated_log.write("Losses: " + str(trials.losses()) + "\n")
    #aggregated_log.write("Statuses: " + str(trials.statuses()) + "\n")

    #aggregated_log.close()

    for p in workers:
        p.terminate()