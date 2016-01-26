from spectral_dagger import sample_episode
from spectral_dagger.tests.conftest import make_test_display
from spectral_dagger.envs import LinearMarkovChain, GridWorld
from spectral_dagger.mdp import TD, PolicyIteration, ValueIteration


def test_simple(display=False):
    display_hook = make_test_display(display)

    # 1 action mdp (so a markov chain), no policy
    mdp = LinearMarkovChain(4)
    sample_episode(mdp, hook=display_hook)

    # 1 action mdp (so a markov chain), a learning policy
    policy = TD(mdp, alpha=0.1)
    sample_episode(mdp, policy, hook=display_hook)


def test_policy_iteration(display=False):
    display_hook = make_test_display(display)

    env = GridWorld(gamma=0.9)

    alg = PolicyIteration(threshold=0.0001)
    policy = alg.fit(env)

    sample_episode(env, policy, horizon=20, hook=display_hook)


def test_value_iteration(display=False):
    display_hook = make_test_display(display)

    env = GridWorld(gamma=0.9)

    alg = ValueIteration(threshold=0.0001)
    policy = alg.fit(env)

    sample_episode(env, policy, horizon=20, hook=display_hook)
