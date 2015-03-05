import numpy as np


def dagger(
        pomdp, initial_policy, policy_class, expert, beta,
        num_iterations, num_samples, horizon):

    policy = initial_policy

    policies = [initial_policy]

    data = []

    for i in range(num_iterations):
        b = beta.next()

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

                pomdp.execute_action(action)
                obs = pomdp.get_current_observation()

                policy.action_played(action)
                policy.observation_emitted(obs)

                expert.action_played(action)
                expert.observation_emitted(obs)

                trajectory.append((action, obs))

            data.append((trajectory, expert_actions))

        policy = policy_class(
            pomdp.actions, pomdp.observations, data)

        policies.append(policy)

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
        ['x', ' ', ' ', 'A', 'x'],
        ['x', ' ', ' ', ' ', 'x'],
        ['x', 'G', ' ', ' ', 'x'],
        ['x', 'x', 'x', 'x', 'x']]
    )

    num_colors = 1

    pomdp = grid_world.ColoredGridWorld(num_colors, world)

    # Expert
    discount = 0.99

    print "Training model..."
    pbvi = PBVI()
    expert = pbvi.fit(pomdp, discount, m=4, n=20)
    trajectory, reward = pomdp.sample_trajectory(
        expert, 20, True, display=1)

    # Dagger
    horizon = 5
    num_dagger_iterations = 2
    num_samples_per_iter = 2000

    policy_class = SpectralPlusClassifier
    beta = itertools.chain([1], iter(lambda: 0, 1))

    # Use random because beta[0] == 1
    initial_policy = RandomPolicy(pomdp.actions, [])

    policies = dagger(
        pomdp, initial_policy, policy_class, expert, beta,
        num_dagger_iterations, num_samples_per_iter, horizon)

    num_test_trajectories = 10

    for i in range(num_test_trajectories):
        trajectory, reward = pomdp.sample_trajectory(
            policies[-1], horizon, True, display=1)
