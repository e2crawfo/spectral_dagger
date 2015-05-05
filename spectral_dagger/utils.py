import numpy as np


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
