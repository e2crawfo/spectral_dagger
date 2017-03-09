import os
import numpy as np
import conllu
import itertools

UNIDEP_PATH = "/data/universal_dep/universal-dependencies-1.2/"


def memoize(f):
    """ Memoization decorator for functions taking one or more arguments. """

    class memodict(dict):
        def __init__(self, f):
            self.f = f

        def __call__(self, *args):
            return self[args]

        def __missing__(self, key):
            ret = self[key] = self.f(*key)
            return ret

    return memodict(f)


def filename2language(filename):
    language = filename
    language = language[language.find('_')+1:]
    if '-' in language:
        language = language[:language.find('-')]
    return language


def filename2dsetname(filename):
    dsetname = filename
    if not dsetname.startswith("UD_"):
        raise ValueError("%s is not a valid filename for a dsetname.")
    return dsetname[3:]


def dset_dirs():
    dir_names = [
        f for f in os.listdir(UNIDEP_PATH)
        if (os.path.isdir(os.path.join(UNIDEP_PATH, f)) and
            f.startswith("UD_"))]
    return sorted(dir_names)


def languages():
    languages = set(filename2language(dn) for dn in dset_dirs())
    return sorted(list(languages))


def n_languages():
    return len(languages())


def dset_names():
    dset_names = set(filename2dsetname(dn) for dn in dset_dirs())
    return sorted(list(dset_names))


def n_dsets():
    return len(dset_names())


def load_data(language, filt, get_all=True):
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
        are returned, or just the dataset whose name exactly matches
        ``language``.

    """
    if language not in languages():
        raise ValueError("No data for language %s." % language)

    _dset_dirs = dset_dirs()

    mapper = filename2language if get_all else filename2dsetname
    tags = [mapper(d) for d in _dset_dirs]

    language_dirs = [
        os.path.join(UNIDEP_PATH, d)
        for t, d in zip(tags, _dset_dirs)
        if t == language]

    data_files = []
    for ld in language_dirs:
        dfs = list(filter(filt, os.listdir(ld)))
        data_files.extend([os.path.join(ld, df) for df in dfs])

    return itertools.chain(
        *[conllu.read_conllu(t) for t in data_files])


def load_data_train(language, get_all=False):
    return load_data(
        language, lambda x: 'train' in x and x.endswith('.conllu'), get_all)


def load_data_dev(language, get_all=False):
    return load_data(
        language, lambda x: 'dev' in x and x.endswith('.conllu'), get_all)


def load_data_test(language, get_all=False):
    return load_data(
        language, lambda x: 'test' in x and x.endswith('.conllu'), get_all)


@memoize
def load_POS_train(language, get_all=False):
    """Return list of lists of universal POS tags."""
    return [
        [w.cpostag for w in sentence.words()]
        for sentence in load_data_train(language, get_all)]


@memoize
def load_POS_dev(language, get_all=False):
    """Return list of lists of universal POS tags."""
    return [
        [w.cpostag for w in sentence.words()]
        for sentence in load_data_dev(language, get_all)]


@memoize
def load_POS_test(language, get_all=False):
    """Return list of lists of universal POS tags."""
    return [
        [w.cpostag for w in sentence.words()]
        for sentence in load_data_test(language, get_all)]


class StatsPOS(object):
    def __init__(self, language, get_all=True, clear=True):
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
        s += ("Standard dev of seq length: "
              "%f\n" % self.stats['train_std_seq_len'])

        s += "Dev " + "*" * 20 + "\n"
        s += "Number of sequences: %d\n" % self.stats['dev_n_seq']
        s += "Mean seq length: %f\n" % self.stats['dev_mean_seq_len']
        s += "Standard dev of seq length: %f\n" % self.stats['dev_std_seq_len']

        s += "Test " + "*" * 20 + "\n"
        s += "Number of sequences: %d\n" % self.stats['test_n_seq']
        s += "Mean seq length: %f\n" % self.stats['test_mean_seq_len']
        s += ("Standard dev of seq length: "
              "%f\n" % self.stats['test_std_seq_len'])

        return s

    def __repr__(self):
        return str(self)


universal_pos = [
    'ADJ',
    'ADP',
    'PUNCT',
    'ADV',
    'AUX',
    'SYM',
    'INTJ',
    'CONJ',
    'X',
    'NOUN',
    'DET',
    'PROPN',
    'NUM',
    'VERB',
    'PART',
    'PRON',
    'SCONJ']


universal_pos_map = {pos: i for i, pos in enumerate(universal_pos)}


def map_nested(nested):
    return [[universal_pos_map[s] for s in seq] for seq in nested]


class SequenceData(object):
    """ Retrieves data for a language, maintaining separation between
        train, dev and test sets. """
    def __init__(self, language, fast=True, get_all=True):
        self.language = language

        if fast:
            pos_dir = os.path.join(UNIDEP_PATH, 'unidep_pos')
            lang_dir = os.path.join(pos_dir, language)
            if not os.path.isdir(lang_dir):
                print("POS data doesn't exist, extracting it...")
                store_pos(language)
                print("Done.")

            lang_dir = os.path.join(pos_dir, language)

            self.train = []
            train = os.path.join(lang_dir, 'train')
            with open(train, 'r') as f:
                for line in iter(f.readline, ''):
                    self.train.append(list(int(i) for i in line.split(',')))

            self.dev = []
            dev = os.path.join(lang_dir, 'dev')
            with open(dev, 'r') as f:
                for line in iter(f.readline, ''):
                    self.dev.append(list(int(i) for i in line.split(',')))

            self.test = []
            test = os.path.join(lang_dir, 'test')
            with open(test, 'r') as f:
                for line in iter(f.readline, ''):
                    self.test.append(list(int(i) for i in line.split(',')))

        else:
            self.train = map_nested(load_POS_train(language, get_all))
            self.dev = map_nested(load_POS_dev(language, get_all))
            self.test = map_nested(load_POS_test(language, get_all))

    @property
    def all(self):
        return self.train + self.dev + self.test


def store_pos(langs=None, get_all=True):
    """ Read Universal Dependency files for supplied languages (or all
        if None is supplied), create separate files which only store the
        POS data, with symbols encoded as integers.

        Loading files with only the POS info is orders of magnitude faster
        than loading the original files with all the info.

    """
    pos_dir = os.path.join(UNIDEP_PATH, 'unidep_pos')

    if not os.path.isdir(pos_dir):
        os.makedirs(pos_dir)

    if langs is None:
        langs = languages()
    if isinstance(langs, str):
        langs = [langs]

    for language in langs:
        sd = SequenceData(language, fast=False, get_all=get_all)

        lang_dir = os.path.join(pos_dir, language)
        if not os.path.isdir(lang_dir):
            os.makedirs(lang_dir)

        train = os.path.join(lang_dir, 'train')
        with open(train, 'w') as f:
            for seq in sd.train:
                f.write(str(seq)[1:-1] + '\n')
        dev = os.path.join(lang_dir, 'dev')
        with open(dev, 'w') as f:
            for seq in sd.dev:
                f.write(str(seq)[1:-1] + '\n')
        test = os.path.join(lang_dir, 'test')
        with open(test, 'w') as f:
            for seq in sd.test:
                f.write(str(seq)[1:-1] + '\n')


def pos2spice(langs=None, get_all=True):
    """ Convert the POS data (in its original Universal Dependency format)
        to the spice format.

    """
    pos_dir = os.path.join(UNIDEP_PATH, 'unidep_pos_spice')

    if not os.path.isdir(pos_dir):
        os.makedirs(pos_dir)

    if langs is None:
        langs = languages()
    if isinstance(langs, str):
        langs = [langs]

    n_symbols = len(universal_pos)

    for language in langs:
        print("Converting data for %s..." % language)
        sd = SequenceData(language, fast=False, get_all=get_all)

        lang_dir = os.path.join(pos_dir, language)
        if not os.path.isdir(lang_dir):
            os.makedirs(lang_dir)

        write_spice(os.path.join(lang_dir, 'train.spice'), sd.train, n_symbols)
        write_spice(os.path.join(lang_dir, 'dev.spice'), sd.dev, n_symbols)
        write_spice(os.path.join(lang_dir, 'test.spice'), sd.test, n_symbols)


def write_spice(filename, sequences, n_symbols):
    with open(filename, 'w') as f:
        f.write('%s %s\n' % (len(sequences), n_symbols))
        for seq in sequences:
            f.write(str(len(seq)))
            for s in seq:
                f.write(' ' + str(s))
            f.write('\n')
