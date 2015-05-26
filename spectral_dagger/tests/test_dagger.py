from spectral_dagger.dagger import dagger, StateClassifier
from spectral_dagger.dagger import po_dagger, BeliefStateClassifier
from spectral_dagger.spectral_wfa import SpectralClassifier
from spectral_dagger.value_iteration import ValueIteration
import spectral_dagger.grid_world as grid_world
from spectral_dagger.pbvi import PBVI

import numpy as np


import itertools
from sklearn.svm import SVC


def test_dagger(display=False):
    # Define environment
    world_map = np.array([
        ['x', 'x', 'x', 'x', 'x'],
        ['x', ' ', ' ', 'S', 'x'],
        ['x', ' ', ' ', ' ', 'x'],
        ['x', 'G', ' ', ' ', 'x'],
        ['x', 'x', 'x', 'x', 'x']]
    )

    mdp = grid_world.GridWorld(world_map)

    print "Computing expert policy..."
    alg = ValueIteration()
    expert = alg.fit(mdp)

    mdp.sample_trajectory(
        expert, horizon=5, reset=True, display=display)

    learning_alg = StateClassifier(SVC)
    beta = itertools.chain([1], iter(lambda: 0, 1))

    print "Running DAgger..."
    policies = dagger(
        mdp, expert, learning_alg, beta,
        n_iterations=5, n_samples_per_iter=500,
        horizon=5)

    print "Testing final policy returned by DAgger..."
    for i in range(10):
        mdp.sample_trajectory(
            policies[-1], horizon=10, reset=True, display=display)


def test_po_dagger(display=False):
    # Define environment
    world_map = np.array([
        ['x', 'x', 'x', 'x', 'x'],
        ['x', ' ', ' ', ' ', 'x'],
        ['x', ' ', ' ', ' ', 'x'],
        ['x', 'G', ' ', ' ', 'x'],
        ['x', 'x', 'x', 'x', 'x']]
    )

    pomdp = grid_world.EgoGridWorld(n_colors=1, world_map=world_map)

    print "Computing expert policy..."
    alg = PBVI(m=4, n=20)
    expert = alg.fit(pomdp)
    pomdp.sample_trajectory(
        expert, 20, True, display=display)

    learning_alg = BeliefStateClassifier(SVC)
    beta = itertools.chain([1], iter(lambda: 0, 1))

    print "Running DAgger..."
    policies = po_dagger(
        pomdp, expert, learning_alg, beta,
        n_iterations=2, n_samples_per_iter=1000,
        horizon=3)

    print "Testing final policy returned by DAgger..."
    for i in range(10):
        pomdp.sample_trajectory(
            policies[-1], horizon=10, reset=True, display=display)


def test_spectral_dagger(display=False):
    # Define environment
    world_map = np.array([
        ['x', 'x', 'x', 'x', 'x'],
        ['x', ' ', ' ', ' ', 'x'],
        ['x', ' ', ' ', ' ', 'x'],
        ['x', 'G', ' ', ' ', 'x'],
        ['x', 'x', 'x', 'x', 'x']]
    )

    pomdp = grid_world.EgoGridWorld(n_colors=1, world_map=world_map)

    print "Computing expert policy..."
    alg = PBVI(m=4, n=20)
    expert = alg.fit(pomdp)
    pomdp.sample_trajectory(
        expert, horizon=5, reset=True, display=display)

    learning_alg = SpectralClassifier(SVC)
    beta = itertools.chain([1], iter(lambda: 0, 1))

    print "Running DAgger..."
    policies = po_dagger(
        pomdp, expert, learning_alg, beta,
        n_iterations=2, n_samples_per_iter=1000,
        horizon=3)

    print "Testing final policy returned by DAgger..."
    for i in range(10):
        pomdp.sample_trajectory(
            policies[-1], horizon=3, reset=True, display=display)
