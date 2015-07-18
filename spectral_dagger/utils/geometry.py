import numpy as np
import operator


def is_numeric(x):
    return isinstance(x, float) or isinstance(x, int)


class Position(object):
    """
    A mix between a 1-D ndarray of size 2 and a tuple of length 2. Supports
    the convenient algebraic manipulation of the former, and the hashability,
    immutability, and ability to act as an index of the latter.
    """

    def __init__(self, x, y=None):

        if is_numeric(x) and is_numeric(y):
            self.position = np.array([x, y])
        else:
            try:
                self.position = np.array(x, copy=True).flatten()

                assert y is None
                assert self.position.size == 2
                assert self.position.ndim == 1
            except:
                raise NotImplementedError()

        self.position.flags.writeable = False

    def __eq__(self, other):
        try:
            return all(self.position == other.position)
        except:
            try:
                return all(self.position == other)
            except:
                raise NotImplementedError()

    def __add__(self, other):
        try:
            new_position = self.position + other
            return Position(new_position)
        except:
            raise NotImplementedError()

    def __radd__(self, other):
        try:
            new_position = other + self.position
            return Position(new_position)
        except:
            raise NotImplementedError()

    def __iadd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        try:
            new_position = self.position - other
            return Position(new_position)
        except:
            raise NotImplementedError()

    def __rsub__(self, other):
        try:
            new_position = other - self.position
            return Position(new_position)
        except:
            raise NotImplementedError()

    def __isub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        try:
            new_position = self.position * other
            return Position(new_position)
        except:
            raise NotImplementedError()

    def __rmul__(self, other):
        try:
            new_position = other * self.position
            return Position(new_position)
        except:
            raise NotImplementedError()

    def __imul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        try:
            new_position = self.position / other
            return Position(new_position)
        except:
            raise NotImplementedError()

    def __rdiv__(self, other):
        try:
            new_position = other / self.position
            return Position(new_position)
        except:
            raise NotImplementedError()

    def __idiv__(self, other):
        return self.__div__(other)

    def __neg__(self):
        return Position(-self.position)

    def __getitem__(self, index):
        try:
            val = self.position[index]
        except:
            raise KeyError("Invalid key %s used to index %s." % (index, self))

        return val

    def __hash__(self):
        return hash(tuple(self))

    def __iter__(self):
        return iter(self.position)

    def __array__(self):
        return self.position.copy()

    def __str__(self):
        return "<%f, %f>" % tuple(self.position)

    def __repr__(self):
        return str(self)


class Shape(object):
    def __init__(self):
        raise NotImplementedError()


class Rectangle(Shape):
    def __init__(self, s, centre=None, top_left=None, closed=False):
        if centre is not None and top_left is not None:
            raise ValueError(
                "Cannot supply both a centre and a "
                "top-left corner to Rectangle.")

        if centre is None and top_left is None:
            raise ValueError(
                "Must supply either a centre or a top_left corner "
                "to Rectangle.")

        self.s = Position(s)
        assert self.s[0] > 0 and self.s[1] > 0, "Side lengths must be > 0"

        if centre is not None:
            centre = Position(centre)
            top_left = centre - self.s / 2.0

        self.top_left = Position(top_left)
        self.closed = closed

    def __contains__(self, pos):
        op = operator.le if self.closed else operator.lt

        inside = (
            op(self.top_left[0], pos[0])
            and op(pos[0], self.top_left[0] + self.s[0]))

        inside &= (
            op(self.top_left[1], pos[1])
            and op(pos[1], self.top_left[1] + self.s[1]))

        return inside

    def __str__(self):
        s = "<Rectangle. top_left: %s, s: %s>" % (self.top_left, self.s)
        return s

    def __repr__(self):
        return str(self)


class Circle(Shape):
    def __init__(self, r, centre, closed=False):
        self.centre = Position(centre)
        self.r = r
        assert self.r > 0, "Radius must be > 0."

        self.closed = closed

    def __contains__(self, pos):
        op = operator.le if self.closed else operator.lt
        return op(np.linalg.norm(pos - self.centre), self.r)

    def __str__(self):
        s = "<Circle. centre: %s, r: %s>" % (self.centre, self.r)
        return s

    def __repr__(self):
        return str(self)
