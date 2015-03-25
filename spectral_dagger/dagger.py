import numpy as np


def dagger(
        pomdp, initial_policy, policy_class, expert, beta,
        num_iterations, num_samples, horizon):

    print "Running DAgger..."

    policy = initial_policy

    policies = [initial_policy]

    data = []

    for i in range(num_iterations):
        b = beta.next()

        print "Starting DAgger iteration ", i
        print "Beta is ", b

        for j in range(num_samples):
            pomdp.reset()
            expert.reset()
            policy.reset()

            trajectory = []
            expert_actions = []

            for t in range(horizon):
                expert_action = expert.get_action()
                expert_actions.append(expert_action)

                if np.random.random() < b:
                    action = expert_action
                else:
                    action = policy.get_action()

                obs, _ = pomdp.execute_action(action)

                policy.update(action, obs)
                expert.update(action, obs)

                trajectory.append((action, obs))

            data.append((trajectory, expert_actions))

        policy = policy_class(
            pomdp.actions, pomdp.observations, data)

        policies.append(policy)

    print "Done DAgger."

    return policies

if __name__ == "__main__":
    import grid_world
    from pbvi import PBVI
    from spectral_wfa import SpectralPlusClassifier
    from policy import RandomPolicy

    import itertools

    # POMDP
    world = np.array([
        ['x', 'x', 'x', 'x', 'x'],
        ['x', ' ', ' ', ' ', 'x'],
        ['x', ' ', ' ', ' ', 'x'],
        ['x', 'G', ' ', ' ', 'x'],
        ['x', 'x', 'x', 'x', 'x']]
    )

    num_colors = 1

    print "Computing expert policy..."
    pomdp = grid_world.ColoredGridWorld(num_colors, world)

    # Expert
    discount = 0.99

    print "Training model..."
    pbvi = PBVI()
    expert = pbvi.fit(pomdp, discount, m=4, n=20)
    # trajectory, reward = pomdp.sample_trajectory(
    #     expert, 20, True, display=1)

    # Dagger
    dagger_horizon = 3
    num_dagger_iterations = 2
    num_samples_per_iter = 10000

    policy_class = SpectralPlusClassifier
    beta = itertools.chain([1], iter(lambda: 0, 1))

    # Use random because beta[0] == 1
    initial_policy = RandomPolicy(pomdp.actions, [])

    policies = dagger(
        pomdp, initial_policy, policy_class, expert, beta,
        num_dagger_iterations, num_samples_per_iter,
        dagger_horizon)

    num_test_trajectories = 10
    test_horizon = 10

    world = np.array([
        ['x', 'x', 'x', 'x', 'x'],
        ['x', ' ', ' ', 'A', 'x'],
        ['x', ' ', ' ', ' ', 'x'],
        ['x', 'G', ' ', ' ', 'x'],
        ['x', 'x', 'x', 'x', 'x']]
    )

    pomdp = grid_world.ColoredGridWorld(num_colors, world)

    for i in range(num_test_trajectories):
        trajectory, reward = pomdp.sample_trajectory(
            policies[-1], test_horizon, True, display=1)
