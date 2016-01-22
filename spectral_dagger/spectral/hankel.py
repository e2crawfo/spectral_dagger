"""
Functions for constructing bases for Hankel matrices and
estimating Hankel matrices from data.

When constructing Hankel matrices, scipy.sparse.lil_matrices are used
because they are most efficient for building-up matrices, and the
results are returned as scipy.sparse.csr_matrices, as these are most
efficient when manipulating matrices.

"""

import numpy as np
from itertools import product
import heapq
from scipy.sparse import lil_matrix, csr_matrix
from collections import defaultdict


def estimate_hankels(data, basis, observations, estimator):
    prefix_dict, suffix_dict = basis

    size_P = len(prefix_dict)
    size_S = len(suffix_dict)

    hankel = lil_matrix((size_P, size_S))

    symbol_hankels = {}
    for obs in observations:
        symbol_hankels[obs] = lil_matrix((size_P, size_S))

    hp = lil_matrix((size_P, 1))
    hs = lil_matrix((1, size_S))

    fill_funcs = {
        "string": fill_string_hankel,
        "prefix": fill_prefix_hankel,
        "substring": fill_substring_hankel}

    try:
        fill_funcs[estimator](
            data, basis, hp, hs, hankel, symbol_hankels)
    except KeyError:
        raise ValueError("Unknown Hankel estimator name: %s." % estimator)

    hankel = csr_matrix(hankel)

    for obs in observations:
        symbol_hankels[obs] = csr_matrix(symbol_hankels[obs])

    return hp, hs, hankel, symbol_hankels


def fill_string_hankel(data, basis, hp, hs, hankel, symbol_hankels):
    n_samples = len(data)
    prefix_dict, suffix_dict = basis

    for seq in data:
        seq = tuple(seq)

        if seq in prefix_dict:
            hp[prefix_dict[seq], 0] += 1.0 / n_samples

        if seq in suffix_dict:
            hs[0, suffix_dict[seq]] += 1.0 / n_samples

        for i in range(len(seq)+1):
            prefix = tuple(seq[:i])

            if prefix not in prefix_dict:
                continue

            suffix = tuple(seq[i:])

            if suffix in suffix_dict:
                hankel[
                    prefix_dict[prefix],
                    suffix_dict[suffix]] += 1.0 / n_samples

            if suffix:
                o = suffix[0]
                suffix = tuple(suffix[1:])

                if suffix in suffix_dict:
                    symbol_hankel = symbol_hankels[o]
                    symbol_hankel[
                        prefix_dict[prefix],
                        suffix_dict[suffix]] += 1.0 / n_samples


def fill_prefix_hankel(data, basis, hp, hs, hankel, symbol_hankels):
    n_samples = len(data)
    prefix_dict, suffix_dict = basis

    for seq in data:
        for i in range(len(seq)+1):
            prefix = tuple(seq[:i])

            if prefix in prefix_dict:
                hp[prefix_dict[prefix], 0] += 1.0 / n_samples

            if prefix in suffix_dict:
                hs[0, suffix_dict[prefix]] += 1.0 / n_samples

            if prefix not in prefix_dict:
                continue

            for j in range(i, len(seq)+1):
                suffix = tuple(seq[i:j])

                if suffix in suffix_dict:
                    hankel[
                        prefix_dict[prefix],
                        suffix_dict[suffix]] += 1.0 / n_samples

                if suffix:
                    o = suffix[0]
                    suffix = tuple(suffix[1:])

                    if suffix in suffix_dict:
                        symbol_hankel = symbol_hankels[o]
                        symbol_hankel[
                            prefix_dict[prefix],
                            suffix_dict[suffix]] += 1.0 / n_samples


def fill_substring_hankel(data, basis, hp, hs, hankel, symbol_hankels):
    n_samples = len(data)
    prefix_dict, suffix_dict = basis

    for seq in data:
        for i in range(len(seq)+1):  # substring start positions
            for j in range(i, len(seq)+1):  # suffix start positions
                prefix = tuple(seq[i:j])

                if prefix in prefix_dict:
                    hp[prefix_dict[prefix], 0] += 1.0 / n_samples

                if prefix in suffix_dict:
                    hs[0, suffix_dict[prefix]] += 1.0 / n_samples

                if prefix not in prefix_dict:
                    continue

                for k in range(j, len(seq)+1):  # substring end positions
                    suffix = tuple(seq[j:k])

                    if suffix in suffix_dict:
                        hankel[
                            prefix_dict[prefix],
                            suffix_dict[suffix]] += 1.0 / n_samples

                    if suffix:
                        o = suffix[0]
                        suffix = tuple(suffix[1:])

                        if suffix in suffix_dict:
                            symbol_hankel = symbol_hankels[o]
                            symbol_hankel[
                                prefix_dict[prefix],
                                suffix_dict[suffix]] += 1.0 / n_samples


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
        top_k_prefix.extend(prefix_counts.keys())

    top_k_suffix = [()]

    if len(suffix_counts) > k:
        top_k_suffix.extend(
            heapq.nlargest(k, suffix_counts, key=suffix_counts.get))
    else:
        top_k_suffix.extend(suffix_counts.keys())

    if square:
        min_length = min(len(top_k_prefix), len(top_k_suffix))
        top_k_prefix = top_k_prefix[:min_length]
        top_k_suffix = top_k_suffix[:min_length]

    top_k_prefix.sort(key=lambda x: len(x))
    top_k_suffix.sort(key=lambda x: len(x))

    prefix_dict = {item: index for (index, item) in enumerate(top_k_prefix)}
    suffix_dict = {item: index for (index, item) in enumerate(top_k_suffix)}

    return prefix_dict, suffix_dict


def fixed_length_basis(observations, length, with_empty=True):
    """ Returns basis consisting of all sequences of a given length.

    Parameters
    ----------
    observations: list
        All possible observations.
    length: int > 0
        The length of the sequences in the basis.
    with_empty: bool
        Whether to include the empty string in the basis.

    """
    basis = {}

    if with_empty:
        basis[()] = 0

    for seq in product(*[observations for i in range(length)]):
        basis[seq] = len(basis)

    return basis, basis.copy()


def _true_probability_for_hmm(hmm, string, estimator, length):
    if estimator == 'string':
        prob = (
            hmm.get_seq_prob(string)
            if len(string) == length
            else 0)

    elif estimator == 'prefix':
        prob = (
            hmm.get_seq_prob(string)
            if len(string) <= length
            else 0)

    elif estimator == 'substring':
        prob = (
            hmm.get_subsequence_expectation(string, length)
            if len(string) <= length
            else 0)
    else:
        raise ValueError(
            "Unknown Hankel estimator: %s." % estimator)

    return prob


def true_hankel_for_hmm(hmm, basis, length, estimator='string', full=False):
    """ Construct the true Hankel matrix for a given HMM. """

    prefix_dict, suffix_dict = basis

    size_P = len(prefix_dict)
    size_S = len(suffix_dict)

    hankel = lil_matrix((size_P, size_S))

    probabilities = {}

    for prefix, suffix in product(prefix_dict, suffix_dict):
        string = prefix + suffix

        prob = probabilities.get(string)
        if prob is None:
            prob = _true_probability_for_hmm(hmm, string, estimator, length)
            probabilities[string] = prob

        hankel[prefix_dict[prefix], suffix_dict[suffix]] = prob

    if not full:
        return csr_matrix(hankel)
    else:
        hp = lil_matrix((size_P, 1))
        hs = lil_matrix((1, size_S))

        for prefix in prefix_dict:
            hp[prefix_dict[prefix], 0] = _true_probability_for_hmm(
                hmm, prefix, estimator, length)

        for suffix in suffix_dict:
            hs[0, suffix_dict[suffix]] = _true_probability_for_hmm(
                hmm, suffix, estimator, length)

        symbol_hankels = {}
        for obs in hmm.observations:
            symbol_hankels[obs] = lil_matrix((size_P, size_S))

        strings = product(prefix_dict, hmm.observations, suffix_dict)
        for prefix, o, suffix in strings:
            string = prefix + (o,) + suffix

            prob = probabilities.get(string)
            if prob is None:
                prob = _true_probability_for_hmm(
                    hmm, string, estimator, length)
                probabilities[string] = prob

            symbol_hankels[o][prefix_dict[prefix], suffix_dict[suffix]] = prob

        for obs in hmm.observations:
            symbol_hankels[obs] = csr_matrix(symbol_hankels[obs])

        return hp, hs, hankel, symbol_hankels
