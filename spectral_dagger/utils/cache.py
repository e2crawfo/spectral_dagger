import numpy as np
import logging
import os
import six.moves.cPickle as pickle
import hashlib
import functools


logger = logging.getLogger(__name__)


def verbose_print(obj, verbosity, threshold=1.0):
    if verbosity >= threshold:
        print obj


def cached(directory, version=None, verbosity=0.0):
    vprint = functools.partial(verbose_print, verbosity=verbosity)
    if not os.path.isdir(directory):
        os.makedirs(directory)

    def cache_decorator(original_func):
        def new_func(**kwargs):
            vprint("Trying to load...")

            key_args = kwargs.copy()
            if version is not None and 'version' not in key_args:
                key_args['version'] = version

            key = get_cache_key(**key_args)
            filename = os.path.join(directory, key)
            try:
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                vprint("Loaded successfully.")
            except Exception:
                vprint("Loading failed, calling function.")
                data = original_func(**kwargs)
                with open(filename, 'wb') as f:
                    pickle.dump(
                        data, f, protocol=pickle.HIGHEST_PROTOCOL)
            return data

        return new_func
    return cache_decorator


# From nengo 2.0
class Fingerprint(object):
    """Fingerprint of an object instance.

    A finger print is equal for two instances if and only if they are of the
    same type and have the same attributes.

    The fingerprint will be used as identification for caching.

    Parameters
    ----------
    obj : object
        Object to fingerprint.

    """
    __slots__ = ['fingerprint']

    def __init__(self, obj):
        self.fingerprint = hashlib.sha1()
        try:
            self.fingerprint.update(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL))
        except (pickle.PicklingError, TypeError) as err:
            raise ValueError("Cannot create fingerprint: {msg}".format(
                msg=str(err)))

    def __str__(self):
        return self.fingerprint.hexdigest()


PY2 = True


# Based on _get_cache_key from nengo 2.0
def get_cache_key(**kwargs):
    h = hashlib.sha1()

    keys = sorted(kwargs.keys())
    for k in keys:
        h.update(k)
        v = kwargs[k]

        if isinstance(v, np.ndarray):
            h.update(np.ascontiguousarray(v).data)
        else:
            if PY2:
                h.update(str(Fingerprint(v)))
            else:
                h.update(str(Fingerprint(v)).encode('utf-8'))

    return h.hexdigest()
