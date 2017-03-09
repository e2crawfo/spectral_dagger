import numpy as np

from spectral_dagger.tests.conftest import make_test_display
from spectral_dagger.pomdp import PBVI
from spectral_dagger.envs import EgoGridWorld


def test_pbvi(display=False):
    display_hook = make_test_display(display)

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

        print("Training model...")
        alg = PBVI()
        policy = alg.fit(pomdp)

        pomdp.sample_episode(policy, horizon=20, hook=display_hook)
