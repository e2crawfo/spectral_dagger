import abc
import numpy as np
import time

from spectral_dagger.utils import default_rng


class SpectralDaggerObject(object):
    _model_rng = default_rng()
    _simulation_rng = default_rng()

    @property
    def model_rng(self):
        return self._model_rng

    @model_rng.setter
    def model_rng(self, rng):
        if rng is None:
            self._model_rng = SpectralDaggerObject._model_rng
        elif isinstance(rng, np.random.RandomState):
            self._model_rng = rng
        else:
            raise ValueError(
                "``rng`` must be None or an instance of np.random.RandomState")

    @property
    def rng(self):
        return self._simulation_rng

    @rng.setter
    def rng(self, rng):
        if rng is None:
            self._simulation_rng = SpectralDaggerObject._simulation_rng
        elif isinstance(rng, np.random.RandomState):
            self._simulation_rng = rng
        else:
            raise ValueError(
                "``rng`` must be None or an instance of np.random.RandomState")


def set_model_rng(rng):
    SpectralDaggerObject._model_rng = default_rng(rng)


def set_sim_rng(rng):
    SpectralDaggerObject._simulation_rng = default_rng(rng)


class Space(object):
    """ A space of objects.

    Each element of ``dimensions`` corresponds to a dimension of the space.
    For each discrete dimension, the corresponding list element should be
    a set giving all possible values for the dimension (each element should
    be comparable).  For each continuous dimension, the corresponding list
    element should be a 2-tuple giving the lower and upper bounds for the
    dimension.

    Parameters
    ----------
    dimensions: list or None
        A list of dimension specifications. None corresponds to the space
        containing everything. An empty list correspods to the empty space.
        Can also supply value for a single dimension, not in a list.

    name: string
        A name for the space.

    """
    def __init__(self, dimensions=None, name=""):
        if not dimensions:
            raise ValueError("``dimensions`` must not be empty.")

        if isinstance(dimensions, tuple) or isinstance(dimensions, set):
            dimensions = [dimensions]

        self.dimensions = dimensions

        for d in dimensions:
            if isinstance(d, tuple):
                if not len(d) == 2:
                    raise ValueError(
                        "Tuple specifying bounds of continuous dimension "
                        "must have length 2.")

                if d[0] < d[1]:
                    raise ValueError(
                        "Lower bound must be smaller than upper bound.")

            elif isinstance(d, set):
                if not d:
                    raise ValueError(
                        "Set specifying discrete dimension must not be empty.")

            else:
                raise ValueError(
                    "Dimensions spec. must be either a tuple or a set.")

        self.name = "" if name is None else name

        self._is_degenerate = True
        for d in self.dimensions:
            degenerate_dim = isinstance(d, set) and len(d) <= 1
            degenerate_dim |= isinstance(d, tuple) and d[0] == d[1]

            self._is_degenerate &= degenerate_dim

    def is_degenerate(self):
        return self._is_degenerate

    def validate(self, action):
        """ TODO: implement properly. """
        return action

    def __str__(self):
        return "<Space name: %s, dimensions: %s>" % (
            self.name, len(self.dimensions))

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if len(self.dimensions) != len(other.dimensions):
                return False

            for d, d_prime in zip(self.dimensions, other.dimensions):
                if isinstance(d, tuple):
                    eq = (
                        isinstance(d_prime, tuple)
                        and np.isclose(d[0], d_prime[0])
                        and np.isclose(d[1], d_prime[1]))

                    if not eq:
                        return False

                if isinstance(d, list):
                    if not d == d_prime:
                        return False

            return True

        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self.__eq__(other)

        return NotImplemented


class DiscreteObject(object):
    def __init__(self, id, name=""):
        self.id = id
        self.name = name

    def get_id(self):
        return self.id

    def __str__(self):
        s = "<%s id: %d, dim: %d" % (
            self.__class__, self.get_id(), self.dimension)

        if self.name:
            s += ", name: " + self.name
        return s + ">"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        """ Override the default == behavior. """
        if isinstance(other, self.__class__):
            return self.get_id() == other.get_id()
        elif isinstance(other, int):
            return self.get_id() == other

        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, self.__class__) or isinstance(other, int):
            return not self.__eq__(other)

        return NotImplemented

    def __hash__(self):
        """ Override default behaviour when used as key in dict. """
        return hash(self.get_id())

    def __index__(self):
        return self.get_id()

    def __int__(self):
        return self.get_id()


class Action(DiscreteObject):
    def __init__(self, id, name=""):
        super(Action, self).__init__(id, name)


class Observation(DiscreteObject):
    def __init__(self, id, name=""):
        super(Observation, self).__init__(id, name)


class State(DiscreteObject):
    def __init__(self, id, name=""):
        super(State, self).__init__(id, name)


class Environment(SpectralDaggerObject):
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def action_space(self):
        """ The env accepts actions that are elements of this space. """
        return NotImplementedError()

    @abc.abstractproperty
    def observation_space(self):
        """ The env emits observations that are elements of this space. """
        raise NotImplementedError()

    @abc.abstractmethod
    def in_terminal_state(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def has_terminal_states(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def has_reward(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def reset(self, initial=None):
        """ Reset the environment.

        Should be overriden by subclasses. Called at the beginning of
        an episode. If the environment needs to supply an initial observation,
        it should be returned from this function.

        Parameters
        ----------
        initial: any
            An initial state or initial dist to start the episode from.
        rng: RandomState instance
            RNG to use for the episode.

        """
        raise NotImplementedError()

    def start_episode(self, initial=None):
        """ Prepare the environment for a new episode.

        This function should not be overriden, override ``reset`` instead.
        Called at the beginning of an episode. If the environment needs
        to supply an initial observation, it is returned from this function.

        Parameters
        ----------
        initial: any
            An initial state or initial dist to start the episode from.

        """
        return self.reset()

    @abc.abstractmethod
    def update(self, action=None):
        """ Update the environment given that ``action`` was taken.

        Environments that do not use actions still need to accept an action
        argument, but can just throw it away.

        Returns
        -------
        (observation, reward)

        For envs without reward, just set the reward to 0.

        """
        raise NotImplementedError()

    def end_episode(self):
        pass


class Policy(SpectralDaggerObject):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractproperty
    def action_space(self):
        """ The policy emits actions that are elements of this space. """
        return NotImplementedError()

    @abc.abstractproperty
    def observation_space(self):
        """ The policy takes observations that are elements of this space. """
        raise NotImplementedError()

    @abc.abstractmethod
    def reset(self, initial=None):
        """ Reset the policy.

        Should be overriden by subclasses. Called at the beginning of
        an episode. If the environment needs to supply an initial observation,
        it should be returned from this function.

        Parameters
        ----------
        initial: any
            An initial state or initial dist to start the episode from.

        """
        raise NotImplementedError()

    def start_episode(self, initial_obs=None):
        """ Prepare the environment for a new episode.

        This function should not be overriden, override ``reset`` instead.
        Called at the beginning of an episode. If the environment needs
        to supply an initial observation, it is returned from this function.

        Parameters
        ----------
        initial: any
            An initial state or initial dist to start the episode from.

        """
        self.reset(initial_obs)

    @abc.abstractmethod
    def get_action(self):
        """ Return the action the policy chooses to execute.

        Note that the policy should not "update" itself. In other words, it
        should not act as if the action that it returns is *actually* played.
        That is handled by calling the ``update`` method.

        """
        raise NotImplementedError()

    def update(self, action=None, obs=None, reward=None):
        """ Update policy's internal state.

        Update any internal state given that ``action`` was executed, the
        resulting observation was ``obs``, and the resulting reward was
        ``reward``.

        """
        pass

    def end_episode(self):
        pass


def sample_episode(*args, **kwargs):
    return sample_episodes(1, *args, **kwargs)


def sample_episodes(
        n_eps, env, policy=None, horizon=np.inf,
        reset_env=True, hook=None):
    """ Sample a batch of episodes.

    Parameters
    ----------
    n_eps: int
        The number of episodes to sample.

    env: Environment instance
        The environment to sample episodes from.

    policy: Policy instance or list of Policy instances
        Policy or policies that will learn from the episodes, with ``update``
        called on them each time step. If the env requires actions, then the
        first policy is the behaviour policy, and is used to select actions
        using the ``get_action`` method (in addition to the ``update``).

    horizon: positive int or np.inf
        The maximum length of the episodes. Episodes may terminate earlier if
        the env has terminal states. Cannot be infinite if the env lacks
        terminal states.

    reset_env: bool
        Whether to call ``reset`` on the env at the beginning of each episode.

    hook: function (optional)
        A function that is called every time step with the results from that
        time step. Useful e.g. for logging or displaying.

    """
    if policy is None:
        policies = []
    elif hasattr(policy, "__iter__"):
        policies = list(policy)
    else:
        policies = [policy]

    do_actions = not (
        env.action_space is None or env.action_space.is_degenerate())

    behaviour_policy = None
    if do_actions:
        if not policies:
            raise ValueError(
                "Environment requires actions, but no policy was provided.")

        behaviour_policy = policies[0]

        if not env.action_space == behaviour_policy.action_space:
            raise ValueError(
                "Policy and environment must operate "
                "on the same action space.")

    if horizon is np.inf and not env.has_terminal_states():
        raise ValueError(
            "Must supply a finite horizon to sample with "
            "an environment that lacks terminal states.")

    if horizon < 1:
        raise ValueError("``horizon`` must be a positive number.")

    do_reward = env.has_reward()
    just_obs = not do_reward and not do_actions

    episodes = []

    for ep_idx in range(n_eps):
        if reset_env:
            init = None if reset_env is True else reset_env

            # Handle possibility of an initial observation.
            obs = env.start_episode(init)

        for p in policies:
            p.start_episode(obs)

        if obs is None:
            episode = []
        else:
            episode = [obs] if just_obs else [(None, obs, 0.0)]

        action = None
        terminated = False
        t = 0

        while not terminated:
            if do_actions:
                action = behaviour_policy.get_action()

            result = env.update(action)

            if do_reward:
                obs, reward = result
            else:
                obs, reward = result, 0.0

            for p in policies:
                p.update(action, obs, reward)

            episode.append(
                obs if just_obs else (action, obs, reward))

            terminal = env.in_terminal_state()
            terminated = terminal or horizon and t >= horizon

            if hook:
                hook(env=env, policies=policies, action=action,
                     obs=obs, terminal=terminal, t=t)

            t += 1

        for p in policies:
            p.end_episode()

        env.end_episode()

        episodes.append(episode)

    return episodes


def make_print_hook(delay=0.0):
    def delay_print_hook(env, action, obs, t, *args, **kwargs):
        """ A display hook for use with ``sample_episodes``. """

        print "\n"
        print "t =", t
        print "Action: ", action
        print "Obs: ", obs
        print str(env)

        if delay:
            time.sleep(delay)

    return delay_print_hook


class LearningAlgorithm(SpectralDaggerObject):
    """
    Learning algorithms learn from some of data (e.g. sampled trajectories
    or an explicit model such as an MDP), and yield a policy. The constructor
    should accept parameters of the learning algorithm. The fit function should
    take the data that the algorithm is to learn from and yield a policy.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def fit(self):
        raise NotImplementedError()
