import numpy as np

from spectral_dagger.pbvi import PBVI
from spectral_dagger.grid_world import EgoGridWorld


def do_pbvi(do_plot=False):
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

    n_trials = 3

    means = []
    rewards = []

    colors = [1, 2]

    for n_colors in colors:
        rewards.append([])

        for trial in range(n_trials):
            pomdp = EgoGridWorld(n_colors, world, gamma=0.99)

            print("Training model...")
            alg = PBVI()
            policy = alg.fit(pomdp)

            epsiode = pomdp.sample_episode(
                policy, horizon=20, reset=True, display=True)

            rewards[-1].append(sum(r for a, s, r in epsiode))

            print("Reward using %d colors: %f" % (n_colors, rewards[-1][-1]))

    if do_plot:
        import matplotlib.pyplot as plt

        means = [np.mean(r) for r in rewards]

        plt.plot(list(range(len(colors))), means)
        plt.errorbar(
            list(range(len(colors))), means, yerr=[np.std(r) for r in rewards])
        plt.xlim(-1, len(colors))

        plt.show()


if __name__ == "__main__":
    do_pbvi()
