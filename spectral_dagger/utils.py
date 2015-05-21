import numpy as np
import matplotlib.pyplot as plt


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


class ABLine2D(plt.Line2D):

    """
    Draw a line based on its normal vector, coefs, and an bias.
    Uses the equation:

    coefs[0] * x + coefs[1] * y + bias = 0

    Additional arguments are passed to the <matplotlib.lines.Line2D> constructor.
    """

    def __init__(self, bias, coefs, *args, **kwargs):

        assert len(coefs) == 2
        assert coefs[0] != 0 or coefs[1] != 0

        # get current axes if user has not specified them
        if not 'axes' in kwargs:
            kwargs.update({'axes':plt.gca()})

        ax = kwargs['axes']

        # if unspecified, get the current line color from the axes
        if not ('color' in kwargs or 'c' in kwargs):
            kwargs.update({'color':ax._get_lines.color_cycle.next()})

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
