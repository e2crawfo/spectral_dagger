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

    num_trials = 3

    means = []
    rewards = []

    colors = [1, 2]

    for num_colors in colors:
        rewards.append([])

        for trial in range(num_trials):
            pomdp = EgoGridWorld(num_colors, world, gamma=0.99)

            print "Training model..."
            alg = PBVI()
            policy = alg.fit(pomdp)

            trajectory = pomdp.sample_trajectory(
                policy, horizon=20, reset=True, display=True)

            rewards[-1].append(sum(t[2] for t in trajectory))

            print "Reward using %d colors: %f" % (num_colors, rewards[-1][-1])

    if do_plot:
        import matplotlib.pyplot as plt

        means = [np.mean(r) for r in rewards]

        plt.plot(range(len(colors)), means)
        plt.errorbar(
            range(len(colors)), means, yerr=[np.std(r) for r in rewards])
        plt.xlim(-1, len(colors))

        plt.show()


if __name__ == "__main__":
    do_pbvi()
