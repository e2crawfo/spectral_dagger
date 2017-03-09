"""
Functions for constructing bases for Hankel matrices and
estimating Hankel matrices from data.
When constructing Hankel matrices, scipy.sparse.lil_matrices are used
because they are most efficient for building-up matrices, and the
results are returned as scipy.sparse.csr_matrices (if sparse is true),
as these are most efficient when manipulating matrices.

"""

import numpy as np
from itertools import product
import heapq
from scipy.sparse import lil_matrix, csr_matrix
from collections import defaultdict
import six


def estimate_hankels(data, basis, observations, estimator, sparse=False):
    f = build_frequencies(data, estimator, basis)
    ret = hankels_from_callable(f, basis, observations, sparse=sparse)

    return ret


def set_if_nonzero(f, a, i, j, seq):
    value = f(seq)
    if value != 0.0:
        a[i, j] = value


def hankels_from_callable(f, basis, observations, sparse=False):
    """ f is a callable, accepting sequences and
        returning the value for that sequence. """
    prefix_dict, suffix_dict = basis

    size_P = len(prefix_dict)
    size_S = len(suffix_dict)

    hp = lil_matrix((size_P, 1))
    hs = lil_matrix((1, size_S))
    hankel = lil_matrix((size_P, size_S))
    symbol_hankels = {}
    for o in observations:
        symbol_hankels[o] = lil_matrix((size_P, size_S))

    for prefix, idx in six.iteritems(prefix_dict):
        set_if_nonzero(f, hp, idx, 0, prefix)

    for suffix, idx in six.iteritems(suffix_dict):
        set_if_nonzero(f, hs, 0, idx, suffix)

    for prefix, pidx in six.iteritems(prefix_dict):
        for suffix, sidx in six.iteritems(suffix_dict):
            set_if_nonzero(f, hankel, pidx, sidx, prefix + suffix)

            for o in observations:
                set_if_nonzero(
                    f, symbol_hankels[o], pidx, sidx, prefix + (o,) + suffix)
    if sparse:
        hp = csr_matrix(hp)
        hs = csr_matrix(hs)
        hankel = csr_matrix(hankel)
        symbol_hankels = {
            o: csr_matrix(h) for o, h in six.iteritems(symbol_hankels)}
        return hp, hs, hankel, symbol_hankels
    else:
        symbol_hankels = {
            o: h.toarray() for o, h in six.iteritems(symbol_hankels)}
        return hp.toarray(), hs.toarray(), hankel.toarray(), symbol_hankels


def string_generator(data, f):
    for seq in data:
        seq = tuple(seq)
        yield (tuple(seq), f[seq])


def prefix_generator(data, f):
    for seq in data:
        seq = tuple(seq)
        for end in range(len(seq)+1):
            yield (seq[:end], f[seq])


def substring_generator(data, f):
    for seq in data:
        seq = tuple(seq)
        for end in range(len(seq)+1):
            for start in range(end+1):
                yield (seq[start:end], f[seq])


def build_frequencies(data, estimator, basis=None):
    """ If ``basis`` is not None, then we will only store frequencies for
        sequences that would be required to populate a set of Hankel matrices
        that made use of that basis.

        ``data`` can either be a list of sequences, or a mapping from
        sequences to probabilities.

    """
    generators = {
        'string': string_generator,
        'prefix': prefix_generator,
        'substring': substring_generator}
    try:
        generator = generators[estimator]
    except KeyError:
        raise ValueError("Unknown Hankel estimator name: %s." % estimator)

    if basis is not None:
        prefix_dict, suffix_dict = basis

    assert isinstance(data, (list, dict, np.ndarray))
    mapping = (
        data if isinstance(data, dict)
        else defaultdict(lambda: 1.0/len(data)))

    built_mapping = defaultdict(float)

    for element, value in generator(data, mapping):
        do_store = (
            basis is None or element in built_mapping or
            element in prefix_dict or element in suffix_dict)

        if not do_store:
            for i in range(len(element)+1):
                prefix, suffix = element[:i], element[i:]
                if prefix in prefix_dict:
                    do_store = (
                        suffix in suffix_dict or
                        (suffix and suffix[1:] in suffix_dict))
                    if do_store:
                        break

        if do_store:
            built_mapping[element] += value

    def f(seq):
        return built_mapping.get(seq, 0.0)

    return f


def _true_probability_for_pfa(pfa, string, estimator):
    if estimator == 'string':
        prob = pfa.string_prob(string)
    elif estimator == 'prefix':
        prob = pfa.prefix_prob(string)
    elif estimator == 'substring':
        prob = pfa.get_substring_expectation(string)
    else:
        raise ValueError(
            "Unknown Hankel estimator: %s." % estimator)

    return prob


def true_hankel_for_pfa(pfa, basis, estimator='string', sparse=False):
    """ Construct the true Hankel matrix for a given PFA. """

    def f(seq):
        return _true_probability_for_pfa(pfa, seq, estimator)
    return hankels_from_callable(f, basis, pfa.observations, sparse=sparse)


def estimate_kernel_hankels(data, kernel_info, estimator):
    """
    kernel_info: A KernelInfo instance.

    """
    n_prefix_centers = kernel_info.n_prefix_centers
    n_suffix_centers = kernel_info.n_suffix_centers

    hankel = np.zeros((n_prefix_centers, n_suffix_centers))
    symbol_hankels = [
        np.zeros((n_prefix_centers, n_suffix_centers))
        for i in range(kernel_info.n_obs_centers)]

    hp = np.zeros(n_prefix_centers)

    fill_funcs = {"prefix": fill_prefix_kernel_hankel}

    try:
        fill_func = fill_funcs[estimator]
    except KeyError:
        raise ValueError("Unknown Hankel estimator name: %s." % estimator)

    fill_func(data, kernel_info, hp, hankel, symbol_hankels)

    return hp, hankel, symbol_hankels


def fill_prefix_kernel_hankel(
        data, kernel_info, hp, hankel, symbol_hankels):

    for seq in data:
        seq = np.array(seq)

        for i in range(len(seq)+1):
            prefix = seq[:i]
            k_prefix = kernel_info.eval_prefix_kernel(prefix)

            if k_prefix is None:
                continue

            hp += k_prefix

            for j in range(i, len(seq)+1):
                suffix = seq[i:j]
                k_suffix = kernel_info.eval_suffix_kernel(suffix)

                if k_suffix is not None:
                    hankel += np.outer(k_prefix, k_suffix)

                if suffix.shape[0] > 0:
                    obs = suffix[0]
                    k_obs = kernel_info.eval_obs_kernel(obs)

                    suffix = suffix[1:]
                    k_suffix = kernel_info.eval_suffix_kernel(suffix)

                    if k_obs is not None and k_suffix is not None:
                        X = np.outer(k_prefix, k_suffix)

                        for i, ko in enumerate(k_obs):
                            symbol_hankels[i] += ko * X

    n_samples = len(data)

    hp /= n_samples
    hankel /= n_samples
    symbol_hankels = [sh/n_samples for sh in symbol_hankels]


def fill_mean_hankel(
        data, past_length, future_length, hp, hankel, symbol_hankels):

    for seq in data:
        seq = np.array(seq)

        for i in range(past_length, len(seq)+1):
            past = seq[:i]

            hp += k_prefix

            for j in range(i, len(seq)+1):
                suffix = seq[i:j]
                k_suffix = kernel_info.eval_suffix_kernel(suffix)

                if k_suffix is not None:
                    hankel += np.outer(k_prefix, k_suffix)

                if suffix.shape[0] > 0:
                    obs = suffix[0]
                    k_obs = kernel_info.eval_obs_kernel(obs)

                    suffix = suffix[1:]
                    k_suffix = kernel_info.eval_suffix_kernel(suffix)

                    if k_obs is not None and k_suffix is not None:
                        X = np.outer(k_prefix, k_suffix)

                        for i, ko in enumerate(k_obs):
                            symbol_hankels[i] += ko * X

    n_samples = len(data)

    hp /= n_samples
    hankel /= n_samples
    symbol_hankels = [sh/n_samples for sh in symbol_hankels]


def construct_hankels_with_actions(
        data, basis, actions, observations, max_length=100):
    """ Needs to be fixed like the non-action hankel constructors. """

    prefix_dict, suffix_dict = basis

    size_P = len(prefix_dict)
    size_S = len(suffix_dict)

    hankel = lil_matrix((size_P, size_S))

    symbol_hankels = {}

    for pair in product(actions, observations):
        symbol_hankels[pair] = lil_matrix((size_P, size_S))

    action_sequence_counts = defaultdict(int)

    for seq in data:
        action_seq = tuple([pair[0] for pair in seq])
        action_sequence_counts[action_seq] += 1

    for seq in data:
        action_seq = tuple([pair[0] for pair in seq])
        action_seq_count = action_sequence_counts[action_seq]

        # iterating over prefix start positions
        for i in range(len(seq)+1):
            if i > max_length:
                break

            if len(seq) - i > max_length:
                break

            prefix = tuple(seq[:i])

            if i == len(seq):
                suffix = ()
            else:
                suffix = tuple(seq[i:len(seq)])

            if prefix in prefix_dict and suffix in suffix_dict:
                hankel[
                    prefix_dict[prefix],
                    suffix_dict[suffix]] += 1.0 / action_seq_count

            if i < len(seq):
                a, o = seq[i]

                if i + 1 == len(seq):
                    suffix = ()
                else:
                    suffix = tuple(seq[i+1:len(seq)])

                if prefix in prefix_dict and suffix in suffix_dict:
                    symbol_hankel = symbol_hankels[(a, o)]
                    symbol_hankel[
                        prefix_dict[prefix],
                        suffix_dict[suffix]] += 1.0 / action_seq_count

    hankel = csr_matrix(hankel)

    for pair in product(actions, observations):
        symbol_hankels[pair] = csr_matrix(symbol_hankels[pair])

    return (
        hankel[:, 0], hankel[0, :], hankel, symbol_hankels)


def construct_hankels_with_actions_robust(
        data, basis, actions, observations, max_length=100):
    """ Needs to be fixed like the non-action hankel constructors.

    Uses the robust estimator that accounts for the effects of actions."""

    prefix_dict, suffix_dict = basis

    size_P = len(prefix_dict)
    size_S = len(suffix_dict)

    hankel = lil_matrix((size_P, size_S))

    symbol_hankels = {}

    for pair in product(actions, observations):
        symbol_hankels[pair] = lil_matrix((size_P, size_S))

    prefix_count = defaultdict(int)

    for seq in data:
        seq = tuple(seq)
        for i in range(len(seq)):
            prefix = seq[:i] + (seq[i][0],)
            prefix_count[prefix] += 1

            prefix = seq[:i+1]
            prefix_count[prefix] += 1

    for seq in data:
        seq = tuple(seq)

        estimate = 1.0
        for i in range(len(seq)):
            numer = prefix_count[seq[:i+1]]
            denom = prefix_count[seq[:i] + (seq[i][0],)]
            estimate *= float(numer) / denom

        # iterating over prefix start positions
        for i in range(len(seq)+1):
            if i > max_length:
                break

            if len(seq) - i > max_length:
                break

            prefix = tuple(seq[:i])

            if i == len(seq):
                suffix = ()
            else:
                suffix = tuple(seq[i:len(seq)])

            if prefix in prefix_dict and suffix in suffix_dict:
                hankel[
                    prefix_dict[prefix],
                    suffix_dict[suffix]] = estimate

            if i < len(seq):
                if i + 1 == len(seq):
                    suffix = ()
                else:
                    suffix = tuple(seq[i+1:len(seq)])

                if prefix in prefix_dict and suffix in suffix_dict:
                    symbol_hankel = symbol_hankels[seq[i]]
                    symbol_hankel[
                        prefix_dict[prefix],
                        suffix_dict[suffix]] = estimate

    hankel = csr_matrix(hankel)

    for pair in product(actions, observations):
        symbol_hankels[pair] = csr_matrix(symbol_hankels[pair])

    return (
        hankel[:, 0], hankel[0, :], hankel, symbol_hankels)


def count_strings(data, max_length):
    prefix_counts = defaultdict(int)
    suffix_counts = defaultdict(int)

    for seq in data:
        for i in range(0, len(seq)+1):
            prefix = tuple(seq[:i])
            suffix = tuple(seq[i:])

            if prefix and len(prefix) <= max_length:
                prefix_counts[prefix] += 1

            if suffix and len(suffix) <= max_length:
                suffix_counts[suffix] += 1

    return prefix_counts, suffix_counts


def count_prefixes(data, max_length):
    prefix_counts = defaultdict(int)
    suffix_counts = defaultdict(int)

    for seq in data:
        for i in range(0, len(seq)+1):
            for j in range(i, len(seq)+1):
                prefix = tuple(seq[:i])
                suffix = tuple(seq[i:j])

                if prefix and len(prefix) <= max_length:
                    prefix_counts[prefix] += 1

                if suffix and len(suffix) <= max_length:
                    suffix_counts[suffix] += 1

    return prefix_counts, suffix_counts


def count_substrings(data, max_length):
    prefix_counts = defaultdict(int)
    suffix_counts = defaultdict(int)

    for seq in data:
        for i in range(0, len(seq)+1):
            for j in range(i, len(seq)+1):
                prefix = tuple(seq[:i])
                suffix = tuple(seq[i:j])

                if prefix and len(prefix) <= max_length:
                    prefix_counts[prefix] += 1

                if suffix and len(suffix) <= max_length:
                    suffix_counts[suffix] += 1

    return prefix_counts, suffix_counts


def top_k_basis(data, k, estimator='string', max_length=np.inf, square=True):
    """ Returns top `k` most frequently occuring prefixes and suffixes.

    To be used when estimating hankel matrices for
    string, prefix, or substring probabilities.

    data: list of list
        Each sublist is a list of observations constituting a trajectory.
    k: int
        Maximum size of prefix and suffix lists.
    estimator: string
        The type of estimator we are building a basis for.
        One of  'string', 'prefix', 'substring'.
    square: bool
        Whether to use the same number of prefixes and suffixes.
    max_length: int
        The maximum length of any single prefix or suffix.

    """
    if estimator == 'string':
        prefix_counts, suffix_counts = count_strings(data, max_length)
    elif estimator == 'prefix':
        prefix_counts, suffix_counts = count_prefixes(data, max_length)
    elif estimator == 'substring':
        prefix_counts, suffix_counts = count_substrings(data, max_length)
    else:
        raise ValueError("Unknown Hankel estimator: %s." % estimator)

    top_k_prefix = [()]

    if len(prefix_counts) > k:
        top_k_prefix.extend(
            heapq.nlargest(k, prefix_counts, key=prefix_counts.get))
    else:
        top_k_prefix.extend(list(prefix_counts.keys()))

    top_k_suffix = [()]

    if len(suffix_counts) > k:
        top_k_suffix.extend(
            heapq.nlargest(k, suffix_counts, key=suffix_counts.get))
    else:
        top_k_suffix.extend(list(suffix_counts.keys()))

    if square:
        min_length = min(len(top_k_prefix), len(top_k_suffix))
        top_k_prefix = top_k_prefix[:min_length]
        top_k_suffix = top_k_suffix[:min_length]

    top_k_prefix.sort(key=lambda x: len(x))
    top_k_suffix.sort(key=lambda x: len(x))

    prefix_dict = {item: index for (index, item) in enumerate(top_k_prefix)}
    suffix_dict = {item: index for (index, item) in enumerate(top_k_suffix)}

    return prefix_dict, suffix_dict


def fixed_length_basis(
        observations, prefix_length, suffix_length=None, with_empty=True):
    """ Returns basis consisting of all sequences of a given length.

    Parameters
    ----------
    observations: list
        All possible observations.
    prefix_length: int > 0
        The length of the prefix sequences in the basis.
    suffix_length: int > 0 (optional)
        The length of the suffix sequences in the basis. If omitted,
        ``prefix_length`` is used.
    with_empty: bool
        Whether to include the empty string in the basis.

    """
    prefix_dict = {}
    if with_empty:
        prefix_dict[()] = 0

    for seq in product(*[observations for i in range(prefix_length)]):
        prefix_dict[seq] = len(prefix_dict)

    suffix_length = suffix_length or prefix_length
    if suffix_length == prefix_length:
        suffix_dict = prefix_dict.copy()
    else:
        suffix_dict = {}
        if with_empty:
            suffix_dict[()] = 0
        for seq in product(*[observations for i in range(suffix_length)]):
            suffix_dict[seq] = len(suffix_dict)

    return prefix_dict, suffix_dict
