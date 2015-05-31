import numpy as np
import matplotlib.pyplot as plt

from mdp import MDP


class SingleActionMDP(MDP):
    """
    Create a single-action mdp by merging a multi-action MDP and a policy.
    """

    def __init__(self, mdp, policy):
        self.mdp = mdp
        self.policy = policy
        self.actions = [0]
        self.gamma = mdp.gamma
        self.states = self.mdp.states

    @property
    def name(self):
        return "SingleActionMDP"

    def __str__(self):
        return str(self.mdp)

    def reset(self, state=None):
        self.mdp.reset(state)
        self.policy.reset(self.mdp.current_state)

    def execute_action(self, action=None):
        """ Ignores the given action, uses the action from the policy. """
        a = self.policy.get_action()
        s_prime, r = self.mdp.execute_action(a)
        self.policy.update(a, s_prime, r)
        return s_prime, r

    def in_terminal_state(self):
        return self.mdp.in_terminal_state()

    def has_terminal_states(self):
        return self.mdp.has_terminal_states()

    # TODO: T and R are not really correct, should take policy into account.
    @property
    def T(self):
        return self.mdp.T

    @property
    def R(self):
        return self.mdp.R

    @property
    def current_state(self):
        return self.mdp.current_state


def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


def geometric_sequence(alpha, tau=1.0, start=1):
    i = 0.0
    while True:
        value = alpha**(start+np.floor(i/tau))
        print "value: ", value
        yield value
        i += 1


def make_beta(alpha):
    i = 1
    while True:
        yield (1 - alpha)**(i-1)
        i += 1


def exp_epsilon_schedule(tc):
    tc = float(tc)
    count = [0]

    def e():
        count[0] += 1
        return np.exp(-count[0] / tc)

    return e


def poly_epsilon_schedule(tc):
    tc = float(tc)
    count = [0]

    def e():
        count[0] += 1
        return 1.0 / (count[0] / tc)

    return e


def ndarray_to_string(a):
    s = ""

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            s += a[i, j]
        s += '\n'
    return s


# TODO: fix this up
def contour_animation(linear_gtd, env, env_bounds):
    import matplotlib.pyplot as plt
    from matplotlib import animation
    fig = plt.figure()

    y, x = np.meshgrid(
        np.linspace(0, env_bounds[0], 100),
        np.linspace(0, env_bounds[1], 100),
        indexing='ij')

    @np.vectorize
    def f(y, x):
        return linear_gtd.V((y, x))

    # animation function
    def animate(i):
        # cont = plt.contourf(x, y, z, 25)
        env.sample_trajectory(
            policy=linear_gtd, display=False, reset=True)

        plt.title('Value function. alpha: %f' % linear_gtd.alpha)

        CS = plt.contourf(x, -y, f(y, x))
        return CS

    n_episodes = 100
    anim = animation.FuncAnimation(fig, animate, frames=n_episodes)
    anim.save('animation.mp4')


class ABLine2D(plt.Line2D):

    """
    Draw a line based on its normal vector, coefs, and an bias.
    Uses the equation:

    coefs[0] * x + coefs[1] * y + bias = 0

    Additional arguments are passed to the <matplotlib.lines.Line2D>
    constructor.
    """

    def __init__(self, bias, coefs, *args, **kwargs):

        assert len(coefs) == 2
        assert coefs[0] != 0 or coefs[1] != 0

        # get current axes if user has not specified them
        if 'axes' not in kwargs:
            kwargs.update({'axes': plt.gca()})

        ax = kwargs['axes']

        # if unspecified, get the current line color from the axes
        if not ('color' in kwargs or 'c' in kwargs):
            kwargs.update({'color': ax._get_lines.color_cycle.next()})

        # init the line, add it to the axes
        super(ABLine2D, self).__init__([], [], *args, **kwargs)

        if coefs[1] == 0:
            self._slope = None
            self._bias = -bias / coefs[0]
        else:
            self._slope = -coefs[0] / coefs[1]
            self._bias = -bias / coefs[1]

        ax.add_line(self)

        # cache the renderer, draw the line for the first time
        ax.figure.canvas.draw()
        self._update_lim(None)

        # connect to axis callbacks
        self.axes.callbacks.connect('xlim_changed', self._update_lim)
        self.axes.callbacks.connect('ylim_changed', self._update_lim)

    def _update_lim(self, event):
        """ called whenever axis x/y limits change """

        if self._slope is None:
            y = np.array(self.axes.get_ybound())
            x = self._bias * np.ones(len(y))
        else:
            x = np.array(self.axes.get_xbound())
            y = (self._slope * x) + self._bias

        self.set_data(x, y)
        self.axes.draw_artist(self)
