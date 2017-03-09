import os
from collections import defaultdict
import six
import numpy as np
import matplotlib.pyplot as plt
import rdp

PENDIGITS_PATH = "/data/pendigits/"


class UnipenParser(object):
    """ Parse a unipen file.

    Parameters
    ----------
    filename: str
        Location of data file.
    comment_filter: func
        A function which is used to filter out characters. Only characters for which
        this function returns True (when applied to the comment at the beginning
        of the digit) will be parsed.
    ignore_multisegment: bool
        If true, will ignore digits that have multiple segments (i.e. have a
        pen up/pen down in the middle of the sequence. Otherwise, such digits
        will be included with the pen up/pen down simply removed.

    """
    def __init__(
            self, filename, comment_filter=None,
            ignore_multisegment=True):

        self.filename = filename
        self.comment_filter = comment_filter or (lambda x: True)
        self.ignore_multisegment = ignore_multisegment

    def parse(self):
        with open(self.filename, 'r') as f:
            digits = []  # all digit data in this file

            parsed_digits = filtered_digits = ignored_digits = 0

            parsing_digit = False

            # For the current digit
            data = []
            comments = []
            idx = None
            name = None

            for line in iter(f.readline, ''):
                if parsing_digit:
                    if line.startswith('.COMMENT'):
                        comments.append(line)
                    elif (line.startswith('.PEN_UP') or
                          line.startswith('.PEN_DOWN') or
                          line.startswith('.DT')):
                        continue
                    elif len(line) == 1:
                        if self.comment_filter(comments):
                            digits.append(Digit(data, comments, idx, name))
                            parsed_digits += 1
                        else:
                            filtered_digits += 1

                        parsing_digit = False
                        data = []
                        comments = []
                        idx = None
                        name = None
                    else:
                        # A normal data point
                        data.append([float(i) for i in line.split()])
                elif line.startswith('.SEGMENT '):
                    # The beginning of each entry looks like:
                    # .SEGMENT DIGIT <idx> ? "<name>"
                    # .COMMENT <comment>
                    line = line.split()
                    try:
                        idx = int(line[2])
                        name = line[4]
                        parsing_digit = True
                    except:
                        # We have a multi-segment digit
                        if self.ignore_multisegment:
                            ignored_digits += 1
                            parsing_digit = False
                        else:
                            idx = int(line[2].split('-')[0])
                            name = line[4]
                            parsing_digit = True

            print(("Parsed %d digits, filtered out %d digits, "
                  "ignored %d multi-segment digits." % (
                      parsed_digits, filtered_digits, ignored_digits)))

            return digits


class Digit(object):
    def __init__(self, data, comments, idx, name):
        self.data = data
        self.comments = comments
        self.idx = idx
        self.name = name


def parse_comments(comments):
    assert len(comments) == 1
    comment = comments[0].split()

    digit_id = int(comment[1])
    user_id = int(comment[2])
    digit_idx_for_user = comment[3]

    return digit_id, user_id, digit_idx_for_user


def _process_digit(digit, difference, sample_every, simplify):
    digit_id, user_id, digit_idx_for_user = parse_comments(digit.comments)

    assert isinstance(sample_every, int), (
        "``sample_every`` must be an integer.")
    if sample_every > 1:
        data = np.array(digit.data)[::sample_every, :]
    else:
        data = np.array(digit.data)

    if simplify is True:
        data = simplify_digit(data)
    elif simplify is not None:
        simplify = float(simplify)
        data = rdp.rdp(data, epsilon=simplify)

    data = np.diff(data, axis=0) if difference else data
    return data, digit_id, user_id, digit_idx_for_user


def make_comment_filter(use_digits, use_users):
    def comment_filter(comments):
        digit_id, user_id, _ = parse_comments(comments)
        return user_id in use_users and digit_id in use_digits

    return comment_filter


def get_data(
        difference=False, sample_every=1, ignore_multisegment=False,
        use_users=None, use_digits=None, simplify=None):
    """ Get all the data.

    Parameters
    ----------
    difference: bool
        If True, then rather than returning the absolute location of each
        sample, we return differences between successive samples.

    """
    if use_digits is None:
        use_digits = set(range(10))
    else:
        use_digits = set(use_digits)
    assert max(use_digits) <= 9 and min(use_digits) >= 0

    if use_users is None:
        use_users = set(range(1, 45))
    else:
        use_users = set(use_users)
    assert max(use_users) < 45 and min(use_users) >= 1
    training_users = set(range(1, 30)) & use_users
    testing_users = set(range(31, 45)) & use_users
    testing_users = set([i-30 for i in testing_users])

    data = defaultdict(list)
    labels = defaultdict(list)

    training_filter = make_comment_filter(
        use_users=training_users, use_digits=use_digits)

    training_parser = UnipenParser(
        os.path.join(PENDIGITS_PATH, "pendigits-orig.tra"),
        comment_filter=training_filter,
        ignore_multisegment=ignore_multisegment)
    training_digits = training_parser.parse()

    for digit in training_digits:
        (_data, digit_id, user_id,
         digit_idx_for_user) = _process_digit(digit, difference, sample_every, simplify)

        if digit_id in use_digits:
            data[user_id].append(_data)
            labels[user_id].append(digit_id)

    max_training_id = 30

    testing_filter = make_comment_filter(
        use_users=testing_users, use_digits=use_digits)

    testing_parser = UnipenParser(
        os.path.join(PENDIGITS_PATH, "pendigits-orig.tes"),
        comment_filter=testing_filter,
        ignore_multisegment=ignore_multisegment)
    testing_digits = testing_parser.parse()
    for digit in testing_digits:
        (_data, digit_id, user_id,
         digit_idx_for_user) = _process_digit(digit, difference, sample_every, simplify)
        user_id += max_training_id

        if digit_id in use_digits:
            data[user_id].append(_data)
            labels[user_id].append(digit_id)

    data = sorted(list(six.iteritems(data)), key=lambda x: x[0])
    data = [x[1] for x in data]

    labels = sorted(list(six.iteritems(labels)), key=lambda x: x[0])
    labels = [x[1] for x in labels]

    if difference:
        size = 0.0
        n = 0
        for seqs in data:
            for seq in seqs:
                for s in seq:
                    size += np.linalg.norm(s, ord=2)
                    n += 1
        print(("Average distance: %f" % (size/n)))

    return data, labels


def plot_digit(digit, difference=False):
    """ Digit should be array-like with shape (n_points, 2). """
    digit = np.array(digit)

    if difference:
        digit = np.cumsum(digit, axis=0)

    plt.scatter(digit[:, 0], digit[:, 1], c=np.arange(len(digit)))
    plt.colorbar()
    plt.axis('equal')


def simplify_digit(digit):
    diff = np.diff(digit, axis=0)
    sign = np.sign(diff)
    for i in range(len(sign)-1):
        if sign[i][0] == 0:
            sign[i][0] = sign[i+1][0]
        if sign[i][1] == 0:
            sign[i][1] = sign[i+1][1]

    sign_diff = np.diff(sign, axis=0)

    new_digit = [digit[0]]
    for i, (d, sd) in enumerate(zip(digit[1:], sign_diff)):
        if np.any(sd):
            new_digit.append(d)
    new_digit.append(digit[-1])
    return np.array(new_digit)


if __name__ == "__main__":
    data, labels = get_data(use_users=[1, 2], use_digits=[1])
    epsilon = 5.0
    for d in data[0][:10]:
        plt.subplot(3, 1, 1)
        plot_digit(d)
        plt.subplot(3, 1, 2)
        rdp_d = rdp.rdp(d, epsilon=epsilon)
        plot_digit(rdp_d)
        plt.subplot(3, 1, 3)
        simp_d = simplify_digit(d)
        plot_digit(simp_d)
        plt.show()
