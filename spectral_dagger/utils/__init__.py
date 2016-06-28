from .math import sample_multinomial, normalize, rmse


def indent(s, n):
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
