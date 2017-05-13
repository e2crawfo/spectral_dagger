import os
import tempfile

from spectral_dagger.utils.experiment import ExperimentStore

def test_delete_old():
    with tempfile.TemporaryDirectory() as tmpdirname:
        print(tmpdirname)
        es = ExperimentStore(tmpdirname, max_experiments=3, delete_old=True)
        es.new_experiment('exp1')
        es.new_experiment('exp2')
        es.new_experiment('exp3')

        # Now should delete exp1
        es.new_experiment('exp4')
        print(os.listdir(es.path))

if __name__ == "__main__":
    test_delete_old()

