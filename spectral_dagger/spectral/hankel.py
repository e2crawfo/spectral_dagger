import itertools
import heapq
from scipy.sparse import lil_matrix, csr_matrix
from collections import defaultdict


def construct_hankels(
        data, prefix_dict, suffix_dict, observations, basis_length=100):

    size_P = len(prefix_dict)
    size_S = len(suffix_dict)

    # lil_matrix is best for constructing sparse matrices
    hankel = lil_matrix((size_P, size_S))

    symbol_hankels = {}
    for obs in observations:
        symbol_hankels[obs] = lil_matrix((size_P, size_S))

    n_samples = len(data)

    for seq in data:
        for i in range(len(seq)+1):
            for j in range(i, len(seq)+1):
                prefix = tuple(seq[:i])
                suffix = tuple(seq[i:j])

                if prefix in prefix_dict and suffix in suffix_dict:
                    hankel[
                        prefix_dict[prefix], suffix_dict[suffix]] += 1.0 / n_samples

                if suffix:
                    o = suffix[0]
                    suffix = tuple(list(suffix)[1:])

                    if prefix in prefix_dict and suffix in suffix_dict:
                        symbol_hankel = symbol_hankels[o]
                        symbol_hankel[
                            prefix_dict[prefix],
                            suffix_dict[suffix]] += 1.0 / n_samples

    # csr_matrix is best for manipulating sparse matrices
    hankel = csr_matrix(hankel)

    for obs in observations:
        symbol_hankels[obs] = csr_matrix(symbol_hankels[obs])

    return (
        hankel[:, 0], hankel[0, :], hankel, symbol_hankels)


def construct_hankels_with_actions(
        data, prefix_dict, suffix_dict, actions,
        observations, basis_length=100):

    size_P = len(prefix_dict)
    size_S = len(suffix_dict)

    hankel = lil_matrix((size_P, size_S))

    symbol_hankels = {}

    for pair in itertools.product(actions, observations):
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
            if i > basis_length:
                break

            if len(seq) - i > basis_length:
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

    for pair in itertools.product(actions, observations):
        symbol_hankels[pair] = csr_matrix(symbol_hankels[pair])

    return (
        hankel[:, 0], hankel[0, :], hankel, symbol_hankels)


def construct_hankels_with_actions_robust(
        data, prefix_dict, suffix_dict, actions,
        observations, basis_length=100):
    """ Uses the robust estimator that accounts for the effects of actions."""

    size_P = len(prefix_dict)
    size_S = len(suffix_dict)

    hankel = lil_matrix((size_P, size_S))

    symbol_hankels = {}

    for pair in itertools.product(actions, observations):
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
            if i > basis_length:
                break

            if len(seq) - i > basis_length:
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

    for pair in itertools.product(actions, observations):
        symbol_hankels[pair] = csr_matrix(symbol_hankels[pair])

    return (
        hankel[:, 0], hankel[0, :], hankel, symbol_hankels)


def top_k_basis(data, k):
    """ Returns the top `k` most frequently occuring prefixes and suffixes in the data. """

    prefix_count_dict = defaultdict(int)
    suffix_count_dict = defaultdict(int)

    for seq in data:
        for i in range(0, len(seq)+1):
            for j in range(i, len(seq)+1):
                prefix = tuple(seq[:i])
                suffix = tuple(seq[i:j])

            if prefix:
                prefix_count_dict[prefix] += 1

            if suffix:
                suffix_count_dict[suffix] += 1
    # for seq in data:
    #     for i in range(0, len(seq)+1):
    #         prefix = tuple(seq[:i])

    #         if prefix:
    #             prefix_count_dict[prefix] += 1

    #         if i < len(seq):
    #             suffix = tuple(seq[i:len(seq)])

    #             if suffix:
    #                 suffix_count_dict[suffix] += 1

    top_k_prefix = [()]
    top_k_prefix.extend(
        heapq.nlargest(k, prefix_count_dict, key=prefix_count_dict.get))

    top_k_suffix = [()]
    top_k_suffix.extend(
        heapq.nlargest(k, suffix_count_dict, key=suffix_count_dict.get))

    min_length = min(len(top_k_prefix), len(top_k_suffix))
    top_k_prefix = top_k_prefix[:min_length]
    top_k_suffix = top_k_suffix[:min_length]

    prefix_dict = {item: index for (index, item) in enumerate(top_k_prefix)}
    suffix_dict = {item: index for (index, item) in enumerate(top_k_suffix)}

    return prefix_dict, suffix_dict


def fair_basis(data, k, horizon):
    """Returns a basis with the top k elements at each length.
    Currently assumes all trajectories are the same length."""

    prefixes = set(())
    suffixes = set(())

    for i in range(0, horizon+1):
        prefix_count_dict = defaultdict(int)
        suffix_count_dict = defaultdict(int)

        for seq in data:
            prefix = tuple(seq[:i])

            if prefix:
                prefix_count_dict[prefix] += 1

            if i < len(seq):
                suffix = tuple(seq[i:horizon])

                if suffix:
                    suffix_count_dict[suffix] += 1

        best_prefixes = heapq.nlargest(
            k, prefix_count_dict, key=prefix_count_dict.get)

        for p in best_prefixes:
            prefixes.add(p)

        best_suffixes = heapq.nlargest(
            k, suffix_count_dict, key=suffix_count_dict.get)

        for s in best_suffixes:
            suffixes.add(s)

    while len(prefixes) < len(suffixes):
        t = np.random.randint(len(data))
        i = np.random.randint(horizon)

        prefixes.add(tuple(data[t][:i+1]))

    while len(suffixes) < len(prefixes):
        t = np.random.randint(len(data))
        i = np.random.randint(horizon)

        prefixes.add(tuple(data[t][i:]))

    prefixes = list(prefixes)
    suffixes = list(suffixes)

    prefix_dict = {item: index for (index, item) in enumerate(prefixes)}
    suffix_dict = {item: index for (index, item) in enumerate(suffixes)}

    return prefix_dict, suffix_dict

