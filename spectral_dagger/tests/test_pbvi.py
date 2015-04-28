import numpy as np

from spectral_dagger.pbvi import PBVI
from spectral_dagger.grid_world import EgoGridWorld


def test_pbvi(do_plot=False):
    world = np.array([
        ['x', 'x', 'x', 'x', 'x'],
        ['x', ' ', ' ', ' ', 'x'],
        ['x', ' ', ' ', ' ', 'x'],
        ['x', 'x', 'x', ' ', 'x'],
        ['x', 'G', 'x', ' ', 'x'],
        ['x', ' ', ' ', ' ', 'x'],
        ['x', ' ', ' ', ' ', 'x'],
        ['x', 'x', 'x', 'x', 'x']]
    )

    colors = [1, 2]

    for n_colors in colors:
        pomdp = EgoGridWorld(n_colors, world, gamma=0.99)

        print "Training model..."
        alg = PBVI()
        policy = alg.fit(pomdp)

        pomdp.sample_trajectory(
            policy, horizon=20, reset=True, display=True)
