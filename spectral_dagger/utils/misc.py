import os
from os import path
import shutil
import string
import datetime
from contextlib import contextmanager
import six
import sys
import smtplib
from collections import defaultdict
import dill as pickle
from email import encoders
try:
    from email.utils import formatdate
except ImportError:
    from email.Utils import formatdate
from six.moves.email_mime_text import MIMEText
from six.moves.email_mime_base import MIMEBase
from six.moves.email_mime_multipart import MIMEMultipart
from six.moves.configparser import ConfigParser, NoOptionError
from functools import reduce
from zipfile import ZipFile

_title_width = 80
_title_format = "\n{{0:=<{0}.{0}s}}".format(_title_width)


class ObjectSaver(object):
    def __init__(self, dirname, eager=True):
        self._objects = defaultdict(lambda: {})
        self._counts = defaultdict(int)
        self._dirname = dirname
        self.eager = eager

    def add_object(self, obj, kind, idx=None, **kwargs):
        if idx is None:
            idx = len(self._objects.get(kind, {}))
        assert isinstance(idx, int), "Indices must be integers."

        if self.eager:
            self._save_object(obj, kind, idx, **kwargs)
            self._objects[kind][idx] = None
        else:
            self._objects[kind][idx] = (obj, kwargs)

        return idx

    def n_objects(self, kind=None):
        if kind is None:
            return sum(len(v) for v in list(self._objects.values()))

        return len(self._objects.get(kind, {}))

    def _save_object(self, obj, kind, idx, **kwargs):
        kind_dir = path.join(self._dirname, kind)
        try:
            os.makedirs(kind_dir)
        except OSError:
            pass

        object_file = path.join(kind_dir, str(idx))
        kwargs['__object'] = obj
        with open(object_file, 'w') as f:
            pickle.dump(kwargs, f, protocol=pickle.HIGHEST_PROTOCOL)

    def save(self):
        if self.eager:
            print("ObjectSaver: Do not need to call ``save`` when running in eager mode.")
            return

        for kind, objects in list(self._objects.items()):
            for idx, (obj, kwargs) in list(objects.items()):
                self._save_object(obj, kind, idx, **kwargs)


class ObjectLoader(object):
    def __init__(self, dirname, eager=True):
        self._dirname = dirname

    def load_object(self, kind, idx):
        kind_dir = path.join(self._dirname, kind)
        if not path.isdir(kind_dir):
            raise KeyError("Found no objects of kind {}.".format(kind))

        object_file = path.join(kind_dir, str(idx))
        try:
            with open(object_file, 'r') as f:
                kwargs = pickle.load(f)
        except IOError:
            raise KeyError(
                "Could not find an object of kind {} with index {}.".format(kind, idx))

        obj = kwargs.pop('__object')
        return obj, kwargs

    def indices_for_kind(self, kind):
        kind_path = path.join(self._dirname, kind)
        if not path.exists(kind_path):
            return []
        return sorted([int(i) for i in os.listdir(kind_path)])

    def load_objects_of_kind(self, kind):
        d = {}
        for idx in self.indices_for_kind(kind):
            d[idx] = self.load_object(kind, idx)
        return d

    def n_objects_of_kind(self, kind):
        return len(self.indices_for_kind(kind))


class ZipObjectLoader(ObjectLoader):
    def __init__(self, zipname, eager=True):
        self._zip = ZipFile(zipname, 'r')
        self._zipname = os.path.splitext(os.path.basename(zipname))[0]

    def __enter__(self):
        pass

    def __exit__(self):
        self._zip.close()

    def load_object(self, kind, idx):
        name = '{}/{}/{}'.format(self._zipname, kind, idx)
        z = self._zip.open(name)
        kwargs = pickle.load(z)
        obj = kwargs.pop('__object')
        return obj, kwargs

    def indices_for_kind(self, kind):
        kind_path = path.join(self._zipname, kind)
        indices = []
        for s in self._zip.namelist():
            if s.startswith(kind_path):
                indices.append(int(os.path.basename(s)))
        return sorted(indices)


@contextmanager
def cd(path):
    """ cd into dir on __enter__, cd back on exit. """

    old_dir = os.getcwd()
    os.chdir(path)

    try:
        yield
    finally:
        os.chdir(old_dir)


@contextmanager
def redirect_stdout(f):
    """ ``f`` is a file-like object. """
    sys_stdout = sys.stdout
    sys.stdout = f

    try:
        yield
    finally:
        sys.stdout = sys_stdout


@contextmanager
def redirect_stderr(f):
    """ ``f`` is a file-like object. """
    sys_stderr = sys.stderr
    sys.stderr = f

    try:
        yield
    finally:
        sys.stderr = sys_stderr


@contextmanager
def remove_file(path):
    path = os.path.abspath(path)
    try:
        yield
    finally:
        try:
            os.remove(path)
        except:
            pass


@contextmanager
def remove_tree(path):
    path = os.path.abspath(path)
    try:
        yield
    finally:
        try:
            shutil.rmtree(path)
        except:
            pass


def as_title(title, title_format=_title_format):
    if title_format is None:
        title_format = _title_format

    return title_format.format(title + ' ')


def indent(s, n):
    """ Indent a string.

    Parameters
    ----------
    s: str
        String to indent.
    n: int > 0
        Number of times to indent.

    """
    add_newline = False
    if s[-1] == '\n':
        add_newline = True
        s = s[:-1]

    _indent = " " * (4 * n)
    s = s.replace('\n', '\n' + _indent)
    s = _indent + s
    if add_newline:
        s += '\n'
    return s


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
    key_vals = list(config_dict.items())
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
            lambda y, z: y.replace(z, "_"),
            [date_time_string, ":", " ", "-"])
        labels.append(date_time_string)

    file_name = sep.join(labels)

    if extension:
        if extension[0] != '.':
            extension = '.' + extension

        file_name += extension

    return file_name


def send_email(host, from_addr, password, subject, body,
               to_addr, cc_addr=None, bcc_addr=None, files_to_attach=None, port=465):
    if isinstance(to_addr, str):
        to_addr = [to_addr]

    if isinstance(cc_addr, str):
        cc_addr = [cc_addr]
    elif cc_addr is None:
        cc_addr = []

    if isinstance(bcc_addr, str):
        bcc_addr = [bcc_addr]
    elif bcc_addr is None:
        bcc_addr = []

    if isinstance(files_to_attach, str):
        files_to_attach = [files_to_attach]
    elif files_to_attach is None:
        files_to_attach = []

    # create the message
    msg = MIMEMultipart()
    msg["From"] = from_addr
    msg["Subject"] = subject
    msg["Date"] = formatdate(localtime=True)
    msg.attach(MIMEText(body))

    msg["To"] = ', '.join(to_addr)
    msg["cc"] = ', '.join(cc_addr)
    msg["bcc"] = ', '.join(bcc_addr)

    attachment = MIMEBase('application', "octet-stream")

    for f2a in files_to_attach:
        try:
            header = 'Content-Disposition', 'attachment; filename="%s"' % f2a
            with open(f2a, "rb") as fh:
                data = fh.read()
            attachment.set_payload(data)
            encoders.encode_base64(attachment)
            attachment.add_header(*header)
            msg.attach(attachment)
        except IOError:
            msg = "Error opening attachment file %s" % f2a
            print(msg)
            sys.exit(1)

    emails = to_addr + cc_addr + bcc_addr

    server = smtplib.SMTP_SSL(host, port=port)
    server.login(from_addr, password)
    server.sendmail(from_addr, emails, msg.as_string())
    server.quit()


SEP = '\,'


def send_email_using_cfg(cfg_name, **kwargs):
    """ Any values that are not provided with the function
        call will be extracted from the given config file. """
    cfg = ConfigParser()
    cfg.read(cfg_name)

    for k, v in six.iteritems(kwargs):
        kwargs[k] = str(v)

    required = ['host', 'from_addr', 'password', 'subject', 'body', 'to_addr']

    new_kwargs = {}
    for name in required:
        val = cfg.get('email', name, vars=kwargs)

        SEP_posn = string.find(val, SEP)
        if SEP_posn != -1:
            # String contains the separator.
            val = val.split(SEP)

        new_kwargs[name] = val

    optional = ['cc_addr', 'bcc_addr', 'files_to_attach', 'port']
    for name in optional:
        try:
            val = cfg.get('email', name, vars=kwargs)

            SEP_posn = string.find(val, SEP)
            if SEP_posn != -1:
                # String contains the separator.
                val = val.split(SEP)

            new_kwargs[name] = val
        except NoOptionError:
            pass

    return send_email(**new_kwargs)
