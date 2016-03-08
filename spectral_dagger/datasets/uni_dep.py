import os
import numpy as np
import conllu
import itertools

UNIDEP_PATH = "/data/universal_dep/universal-dependencies-1.2/"


def languages():
    fnames = filter(lambda x: os.path.isdir(x), os.listdir(UNIDEP_PATH))
    languages = set(fn.split('_')[-1].split('-')[0] for fn in fnames)
    return sorted(list(languages))


def n_languages():
    return len(languages())


def datasets():
    fnames = filter(lambda x: os.path.isdir(x), os.listdir(UNIDEP_PATH))
    return sorted([fn.split('_')[-1] for fn in fnames])


def n_datasets():
    return len(datasets())


def load_data(language, filt, get_all=False):
    """ Load Universal Dependency data stored in CoNULL-U format.

    Parameters
    ----------
    language: string
        Language/dataset to load data for.
    filt: function, accepts filename as arg
        Only files with names for which the function
        returns true will be processed.
    get_all: bool
        If ``language`` is the name of a language (as opposed to a
        dataset), then this controls whether all datasets for that language
        are returned, or just the dataset whose name exactly matches the
        language.

    """
    if language not in languages():
        raise ValueError("No data for language %s." % language)

    pred = (lambda l, d: l in d) if get_all else (lambda l, d: l == d)

    language_dirs = [
        os.path.join(UNIDEP_PATH, 'UD_'+d)
        for d in datasets() if pred(language, d)]

    train_files = []
    for ld in language_dirs:
        tf = filter(filt, os.listdir(ld))
        train_files.extend([os.path.join(ld, t) for t in tf])

    return itertools.chain(
        *[conllu.read_conllu(t) for t in train_files])


def load_data_train(language, get_all=False):
    return load_data(
        language, lambda x: 'train' in x and x.endswith('.conllu'), get_all)


def load_data_dev(language, get_all=False):
    return load_data(
        language, lambda x: 'dev' in x and x.endswith('.conllu'), get_all)


def load_data_test(language, get_all=False):
    return load_data(
        language, lambda x: 'test' in x and x.endswith('.conllu'), get_all)


def load_POS_train(language, get_all=False):
    """Return list of lists of universal POS tags."""
    return [
        [w.cpostag for w in sentence.words()]
        for sentence in load_data_train(language, get_all)]


def load_POS_dev(language, get_all=False):
    """Return list of lists of universal POS tags."""
    return [
        [w.cpostag for w in sentence.words()]
        for sentence in load_data_dev(language, get_all)]


def load_POS_test(language, get_all=False):
    """Return list of lists of universal POS tags."""
    return [
        [w.cpostag for w in sentence.words()]
        for sentence in load_data_test(language, get_all)]


class StatsPOS(object):
    def __init__(self, language, get_all=False, clear=True):
        self.language = language
        self.stats = {}

        train = load_POS_train(language, get_all)
        self.stats['train_n_seq'] = len(train)

        lengths = [len(t) for t in train]
        self.stats['train_mean_seq_len'] = np.mean(lengths)
        self.stats['train_std_seq_len'] = np.std(lengths)

        if clear:
            del train

        dev = load_POS_dev(language, get_all)
        self.stats['dev_n_seq'] = len(dev)

        lengths = [len(t) for t in dev]
        self.stats['dev_mean_seq_len'] = np.mean(lengths)
        self.stats['dev_std_seq_len'] = np.std(lengths)

        if clear:
            del dev

        test = load_POS_test(language, get_all)
        self.stats['test_n_seq'] = len(test)

        lengths = [len(t) for t in test]
        self.stats['test_mean_seq_len'] = np.mean(lengths)
        self.stats['test_std_seq_len'] = np.std(lengths)

        if clear:
            del test

    def __str__(self):
        s = "Language: %s\n" % self.language

        s += "Train " + "*" * 20 + "\n"
        s += "Number of sequences: %d\n" % self.stats['train_n_seq']
        s += "Mean seq length: %f\n" % self.stats['train_mean_seq_len']
        s += "Standard dev of seq length: %f\n" % self.stats['train_std_seq_len']

        s += "Dev " + "*" * 20 + "\n"
        s += "Number of sequences: %d\n" % self.stats['dev_n_seq']
        s += "Mean seq length: %f\n" % self.stats['dev_mean_seq_len']
        s += "Standard dev of seq length: %f\n" % self.stats['dev_std_seq_len']

        s += "Test " + "*" * 20 + "\n"
        s += "Number of sequences: %d\n" % self.stats['test_n_seq']
        s += "Mean seq length: %f\n" % self.stats['test_mean_seq_len']
        s += "Standard dev of seq length: %f\n" % self.stats['test_std_seq_len']

        return s

    def __repr__(self):
        return str(self)


universal_pos = [
    u'ADJ',
    u'ADP',
    u'PUNCT',
    u'ADV',
    u'AUX',
    u'SYM',
    u'INTJ',
    u'CONJ',
    u'X',
    u'NOUN',
    u'DET',
    u'PROPN',
    u'NUM',
    u'VERB',
    u'PART',
    u'PRON',
    u'SCONJ']


universal_pos_map = {pos: i for i, pos in enumerate(universal_pos)}


def map_nested(nested):
    return [[universal_pos_map[s] for s in seq] for seq in nested]


class SequenceData(object):
    def __init__(self, language, get_all=False):
        self.language = language

        self.train = map_nested(load_POS_train(language, get_all))
        self.dev = map_nested(load_POS_dev(language, get_all))
        self.test = map_nested(load_POS_test(language, get_all))
