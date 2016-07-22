# Some special HMMs
import numpy as np
import networkx as nx


from spectral_dagger.sequence import HMM


"""
These are from the following paper:
    Tran, D., Kim, M., & Doshi-Velez, F. (2016). Spectral M-estimation
    with Applications to Hidden Markov Models. arXiv preprint arXiv:1603.08815.
"""


def noisy_O(error_prob, n_states):
    O = ((1 - error_prob - error_prob/(n_states-1)) * np.eye(n_states) +
         (error_prob / (n_states-1)) * np.ones((n_states, n_states)))
    return O


def Ring(error_prob=0.4, stop_prob=None):
    init_dist = np.ones(5) / 5
    T = [[0, .5, 0, 0, .5],
         [.9, 0, .1, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [.9, 0, 0, .1, 0]]
    O = ((1 - error_prob - error_prob/5) * np.eye(5) +
         (error_prob / 5) * np.ones((5, 5)))
    return HMM(init_dist, T, O, stop_prob)


def Grid(side_length, error_prob=0.4, stop_prob=None):
    n_states = side_length * side_length
    init_dist = np.ones(n_states) / n_states
    G = nx.grid_2d_graph(side_length, side_length)
    T = nx.linalg.adjacency_matrix(G)
    O = noisy_O(error_prob, n_states)
    return HMM(init_dist, T, O, stop_prob)


def Chain(n_states, reset_prob, error_prob=0.4, stop_prob=None):
    init_dist = np.ones(n_states) / n_states
    T = np.zeros((n_states, n_states))
    T[:-1, 1:] = (1 - reset_prob) * np.eye(n_states-1)
    T[:-1, 0] = reset_prob
    T[-1, 0] = 1.0
    O = noisy_O(error_prob, n_states)
    return HMM(init_dist, T, O, stop_prob)
