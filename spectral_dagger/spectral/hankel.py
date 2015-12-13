import itertools
from scipy.sparse import lil_matrix, csr_matrix
from collections import defaultdict


def construct_hankels(
        data, prefix_dict, suffix_dict, observations, basis_length=100):

    import pdb
    pdb.set_trace()

    size_P = len(prefix_dict)
    size_S = len(suffix_dict)

    # lil_matrix is best for constructing sparse matrices
    hankel = lil_matrix((size_P, size_S))

    symbol_hankels = {}
    for obs in observations:
        symbol_hankels[obs] = lil_matrix((size_P, size_S))

    n_samples = len(data)

    for seq in data:
        # iterating over suffix start positions
        for i in range(len(seq)+1):
            # if i > basis_length:
            #     break

            # if len(seq) - i > basis_length:
            #     break

            prefix = tuple(seq[:i])
            suffix = () if i == len(seq) else tuple(seq[i:])

            if prefix in prefix_dict and suffix in suffix_dict:
                hankel[
                    prefix_dict[prefix], suffix_dict[suffix]] += 1.0 / n_samples

            if i < len(seq):
                o = seq[i]
                suffix = () if i+1 == len(seq) else tuple(seq[i+1:])

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
