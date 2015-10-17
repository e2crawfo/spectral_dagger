import numpy as np


def try_loading_ndarray(filename, load_fail_func, *args, **kwargs):
    """ Try loading an ndarray. On failure, call function and save result.

    Parameters
    ----------
    filename: str
        Name of file to load from/save to. Extension '.npy' will be added
        automatically if it is not already there.
    load_fail_func: function, returns ndarray.
        The function to call if loading fails. Must return an ndarray. The
        returned ndarray will be saved to a file with the same name that we
        tried loading from.
    args and kwargs:
        Arguments pass to load_fail_func if it is called.

    Returns
    -------
    The loaded ndarray if loading succeeded, otherwise returns the array
    returned by load_fail_func.
    """

    if not filename.endswith('.npy'):
        filename = filename + '.npy'

    try:
        a = np.load(filename)
    except IOError:
        a = load_fail_func(*args, **kwargs)
        np.save(filename, a)

    return a
