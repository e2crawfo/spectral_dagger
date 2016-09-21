import os
from collections import defaultdict
import six
import numpy as np
import matplotlib.pyplot as plt

PENDIGITS_PATH = "/data/pendigits/"


class UnipenParser(object):
    """ Parse a unipen file.close(

    Parameters
    ----------
    filename: str
        Location of data file.
    ignore_multisegment: bool
        If true, will ignore digits that have multiple segments (i.e. have a
        pen up/pen down in the middle of the sequence. Otherwise, such digits
        will be included with the pen up/pen down simply removed.

    """
    def __init__(self, filename, ignore_multisegment=True):
        self.filename = filename
        self.ignore_multisegment = ignore_multisegment

    def parse(self):
        with open(self.filename, 'r') as f:
            digits = []  # all digit data in this file

            ignored_digits = 0

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
                        digits.append(Digit(data, comments, idx, name))
                        parsing_digit = False
                        data = []
                        comments = []
                        idx = None
                        name = None
                    else:
                        # A normal data point
                        data.append([float(i) for i in line.split()])
                elif line.startswith('.SEGMENT '):
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

            print("Successfully parsed %d digits, "
                  "ignored %d multi-segment digits." % (
                      len(digits), ignored_digits))

            return digits


class Digit(object):
    def __init__(self, data, comments, idx, name):
        self.data = data
        self.comments = comments
        self.idx = idx
        self.name = name


def _process_digit(digit, difference, sample_every):
    assert len(digit.comments) == 1
    comment = digit.comments[0].split()

    digit_id = int(comment[1])
    user_id = int(comment[2])
    digit_idx_for_user = comment[3]

    assert isinstance(sample_every, int), (
        "``sample_every`` must be an integer.")
    if sample_every > 1:
        data = np.array(digit.data)[::sample_every, :]
    else:
        data = np.array(digit.data)

    data = np.diff(data, axis=0) if difference else data
    return data, digit_id, user_id, digit_idx_for_user


def get_data(difference=False, sample_every=1, ignore_multisegment=False, use_digits=None):
    """ Get all the data.

    Parameters
    ----------
    difference: bool
        If True, then rather than returning the absolute location of each
        sample, we return differences between successive samples.

    """
    training_parser = UnipenParser(
        os.path.join(PENDIGITS_PATH, "pendigits-orig.tra"),
        ignore_multisegment=ignore_multisegment)
    training_digits = training_parser.parse()

    if use_digits is None:
        use_digits = set(range(10))
    else:
        use_digits = set(use_digits)
    assert max(use_digits) <= 9 and min(use_digits) >= 0

    data = defaultdict(list)
    labels = defaultdict(list)

    for digit in training_digits:
        (_data, digit_id, user_id,
         digit_idx_for_user) = _process_digit(digit, difference, sample_every)

        if digit_id in use_digits:
            data[user_id].append(_data)
            labels[user_id].append(digit_id)

    max_training_id = max(data.keys())

    testing_parser = UnipenParser(
        os.path.join(PENDIGITS_PATH, "pendigits-orig.tes"),
        ignore_multisegment=ignore_multisegment)
    testing_digits = testing_parser.parse()
    for digit in testing_digits:
        (_data, digit_id, user_id,
         digit_idx_for_user) = _process_digit(digit, difference, sample_every)
        user_id += max_training_id

        if digit_id in use_digits:
            data[user_id].append(_data)
            labels[user_id].append(digit_id)

    data = sorted(list(six.iteritems(data)), key=lambda x: x[0])
    data = [x[1] for x in data]

    labels = sorted(list(six.iteritems(labels)), key=lambda x: x[0])
    labels = [x[1] for x in labels]

    return data, labels


def plot_digit(digit, difference=False):
    """ Digit should be array-like with shape (n_points, 2). """
    digit = np.array(digit)

    if difference:
        digit = np.cumsum(digit, axis=0)

    plt.scatter(digit[:, 0], digit[:, 1], c=np.arange(len(digit)))
    plt.colorbar()


if __name__ == "__main__":
    data, labels = get_data()
