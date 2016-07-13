import os
import string
import datetime

_title_width = 80
_title_format = "\n{{0:=<{0}.{0}s}}".format(_title_width)


def as_title(title, title_format=_title_format):
    if title_format is None:
        title_format = _title_format

    return title_format.format(title + ' ')


def make_symlink(target, name):
    """ NB: ``target`` is just used as a simple string when creating
    the link. That is, ``target`` is the location of the file we want
    to point to, relative to the location that the link resides.
    It is not the case that the target file is identified, and then
    some kind of smart process occurs to have the link point to that file.

    """
    try:
        os.remove(name)
    except OSError:
        pass

    os.symlink(target, name)


def make_filename(main_title, directory='', config_dict=None, use_time=True,
                  sep='_', kvsep=':', extension='', omit=[]):
    """ Create a filename.

    Parameters
    ----------
    main_title: string
        The main title for the file.
    directory: string
        The directory to write the file to.
    config_dict: dict
        Keys and values that will be added to the filename. Key/value
        pairs are put into the filename by the alphabetical order of the keys.
    use_time: boolean
        Whether to append the current date/time to the filename.
    sep: string
        Separates items in the config dict in the returned filename.
    kvsep: string
        Separates keys from values in the returned filename.
    extension: string
        Appears at end of filename.

    """
    if config_dict is None:
        config_dict = {}
    if directory and directory[-1] != '/':
        directory += '/'

    labels = [directory + main_title]
    key_vals = list(config_dict.iteritems())
    key_vals.sort(key=lambda x: x[0])

    for key, value in key_vals:
        if not isinstance(key, str):
            raise ValueError("keys in config_dict must be strings.")
        if not isinstance(value, str):
            raise ValueError("values in config_dict must be strings.")

        if not str(key) in omit and not hasattr(value, '__len__'):
            labels.append(kvsep.join([key, value]))

    if use_time:
        date_time_string = str(datetime.datetime.now()).split('.')[0]
        date_time_string = reduce(
            lambda y, z: string.replace(y, z, "_"),
            [date_time_string, ":", " ", "-"])
        labels.append(date_time_string)

    file_name = sep.join(labels)

    if extension:
        if extension[0] != '.':
            extension = '.' + extension

        file_name += extension

    return file_name
