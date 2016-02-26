import numpy as np
import itertools
from sklearn.svm import SVC
import pytest

from spectral_dagger.tests.conftest import make_test_display
from spectral_dagger.function_approximation import Identity
from spectral_dagger.imitation_learning import dagger, StateClassifier
from spectral_dagger.imitation_learning import po_dagger, BeliefStateClassifier
from spectral_dagger.spectral import SpectralClassifier
from spectral_dagger.mdp import ValueIteration
from spectral_dagger.envs import GridWorld, EgoGridWorld
from spectral_dagger.pomdp import PBVI


def test_dagger(display=False):
    display_hook = make_test_display(display)

    # Define environment
    world_map = np.array([
        ['x', 'x', 'x', 'x', 'x'],
        ['x', ' ', ' ', 'S', 'x'],
        ['x', ' ', ' ', ' ', 'x'],
        ['x', 'G', ' ', ' ', 'x'],
        ['x', 'x', 'x', 'x', 'x']]
    )

    mdp = GridWorld(world_map)

    print "Computing expert policy..."
    alg = ValueIteration()
    expert = alg.fit(mdp)

    mdp.sample_episode(expert, horizon=5, hook=display_hook)

    predictor = SVC()
    learning_alg = StateClassifier(predictor, Identity(2))
    beta = itertools.chain([1], iter(lambda: 0, 1))

    print "Running DAgger..."
    policies = dagger(
        mdp, expert, learning_alg, beta,
        n_iterations=5, n_samples_per_iter=500, horizon=5)

    print "Testing final policy returned by DAgger..."
    mdp.sample_episodes(10, policies[-1], horizon=10, hook=display_hook)


def test_po_dagger(display=False):
    display_hook = make_test_display(display)

    # Define environment
    world_map = np.array([
        ['x', 'x', 'x', 'x', 'x'],
        ['x', ' ', ' ', ' ', 'x'],
        ['x', ' ', ' ', ' ', 'x'],
        ['x', 'G', ' ', ' ', 'x'],
        ['x', 'x', 'x', 'x', 'x']]
    )

    pomdp = EgoGridWorld(n_colors=1, world_map=world_map)

    print "Computing expert policy..."
    alg = PBVI(m=4, n=20)
    expert = alg.fit(pomdp)
    pomdp.sample_episode(expert, horizon=20, hook=display_hook)

    predictor = SVC()
    learning_alg = BeliefStateClassifier(predictor)
    beta = itertools.chain([1], iter(lambda: 0, 1))

    print "Running DAgger..."
    policies = po_dagger(
        pomdp, expert, learning_alg, beta,
        n_iterations=2, n_samples_per_iter=1000,
        horizon=3)

    print "Testing final policy returned by DAgger..."
    pomdp.sample_episodes(10, policies[-1], horizon=10, hook=display_hook)


@pytest.mark.xfail
def test_spectral_dagger(display=False):
    display_hook = make_test_display(display)

    # Define environment
    world_map = np.array([
        ['x', 'x', 'x', 'x', 'x'],
        ['x', ' ', ' ', ' ', 'x'],
        ['x', ' ', ' ', ' ', 'x'],
        ['x', 'G', ' ', ' ', 'x'],
        ['x', 'x', 'x', 'x', 'x']]
    )

    pomdp = EgoGridWorld(n_colors=1, world_map=world_map)

    print "Computing expert policy..."
    alg = PBVI(m=4, n=20)
    expert = alg.fit(pomdp)

    pomdp.sample_episode(expert, horizon=5, hook=display_hook)

    learning_alg = SpectralClassifier(SVC)
    beta = itertools.chain([1], iter(lambda: 0, 1))

    print "Running DAgger..."
    policies = po_dagger(
        pomdp, expert, learning_alg, beta,
        n_iterations=2, n_samples_per_iter=1000,
        horizon=3)

    print "Testing final policy returned by DAgger..."
    pomdp.sample_episodes(10, policies[-1], horizon=3, hook=display_hook)
