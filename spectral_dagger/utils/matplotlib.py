from __future__ import absolute_import

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

from spectral_dagger import sample_episode


# TODO: fix this up
def contour_animation(linear_gtd, env, env_bounds):
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
        sample_episode(env, linear_gtd)

        plt.title('Value function. alpha: %f' % linear_gtd.alpha)

        CS = plt.contourf(x, -y, f(y, x))
        return CS

    n_episodes = 100
    anim = animation.FuncAnimation(fig, animate, frames=n_episodes)
    anim.save('animation.mp4')


class ABLine2D(plt.Line2D):
    """ Draw a line based on its normal vector, coefs, and an bias.

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

