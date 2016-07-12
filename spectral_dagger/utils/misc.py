_title_width = 80
_title_format = "\n{{0:=<{0}.{0}s}}".format(_title_width)


def as_title(title, title_format=None):
    if title_format is None:
        title_format = _title_format

    return title_format.format(title + ' ')

