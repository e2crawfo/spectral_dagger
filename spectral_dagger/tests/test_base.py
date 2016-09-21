from spectral_dagger.envs import LinearMarkovChain


def test_random_state():
    env = LinearMarkovChain(4, p=0.5, gamma=1.0)

    seed = 10
    other_seed = 11
    assert seed != other_seed

    n_eps = 10

    env.random_state = seed
    episodes1 = env.sample_episodes(n_eps)

    env.random_state = seed
    episodes2 = env.sample_episodes(n_eps)

    env.random_state = other_seed
    episodes3 = env.sample_episodes(n_eps)

    episodes1 = [t for ep in episodes1 for t in ep]
    episodes2 = [t for ep in episodes2 for t in ep]
    episodes3 = [t for ep in episodes3 for t in ep]

    assert all([x == y for x, y in zip(episodes1, episodes2)])
    assert any([x != y for x, y in zip(episodes1, episodes3)])
