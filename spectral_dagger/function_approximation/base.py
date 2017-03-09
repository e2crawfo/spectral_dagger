import numpy as np


class FeatureExtractor(object):
    def __init__(self):
        return NotImplementedError("Cannot instantiate FeatureExtractor.")

    @property
    def n_features(self):
        return self._n_features


class Identity(FeatureExtractor):
    def __init__(self, n_features, intercept=True):
        self._n_features = n_features
        self.intercept = intercept

        if self.intercept:
            self._n_features += 1

    def as_vector(self, s):
        vector = np.array(s, copy=True)

        if self.intercept:
            vector = np.concatenate((vector, [1]))

        if vector.ndim != 1:
            raise ValueError(
                "State %s converts to non-vector array." % s)

        if vector.size != self.n_features:
            raise ValueError(
                "State %s converts to array with %d elements, "
                "expecting $d elements." % (s, vector.size, self.n_features))

        return vector


class OneHot(FeatureExtractor):
    def __init__(self, n_features, intercept=True):
        self._n_features = n_features
        self.intercept = intercept

        if self.intercept:
            self._n_features += 1

    def as_vector(self, s):
        vector = np.zeros(self._n_features)

        if self.intercept:
            vector[-1] = 1.0

        vector[s] = 1.0

        return vector


class RectangularTileCoding(FeatureExtractor):
    def __init__(
            self, n_tilings, extent, origin=None,
            tile_dims=None, tile_counts=None, intercept=True):
        """ Implements axis-aligned rectangluar tile coding.

        Parameters
        ----------
        n_tilings: int
            Number of tilings. Offset from origin for each tiling is
            chosen randomly.

        extent: 1-D numpy array
            Dimensions of the (hyper-)rectangle that the tilings should
            cover. Number of entries in this array determines the
            dimensionality of the tiling. Must be at least 1 entry.

        origin: 1-D numpy array
            Location of the corner of the (hyper-) rectangular region
            that the tilings are covering which has the smallest dimensions
            along every dimension. In 2 dimensions, the bottom-left
            corner, assuming the positive orthant is in the top-right.

        (Note: exactly one of the following two parameters must be supplied)

        tile_dims: optional, 1-D numpy array or float
            Dimensions of each tile. If a float is given, the tiles are
            squares with the given side-length.

        tile_counts: optional, 1-D numpy array or int
            Number of tiles along each dimension. If an int is given, the
            same number of tiles is used along every dimension.

        intercept: boolean
            Whether to include an intercept feature, a feature that always
            has value 1.

        """
        assert n_tilings == int(n_tilings), "'n_tilings' must be an integer."
        self.n_tilings = int(n_tilings)

        dtype = np.dtype('d')
        self.dtype = dtype

        self.extent = np.array(extent, dtype=dtype)
        assert self.extent.ndim == 1, "'extent' must be a vector."

        self.n_dims = self.extent.size

        if origin is None:
            origin = np.zeros(self.n_dims)
        self.origin = np.array(origin, dtype=dtype)
        assert self.origin.ndim == 1
        assert self.origin.size == self.n_dims

        if (tile_dims is None) == (tile_counts is None):
            raise ValueError(
                "Must specify either tile_dims or "
                "tile_counts, but not both.")

        if tile_dims is not None:
            try:
                f = float(tile_dims)
                self.tile_dims = f * np.ones(self.n_dims, dtype=dtype)
            except:
                self.tile_dims = np.array(tile_dims, dtype=dtype)

            assert self.tile_dims.ndim == 1
            assert self.tile_dims.size == self.n_dims
            assert all(self.tile_dims > 0), (
                "Detected non-positive tile_dims")
            assert any(self.tile_dims < self.extent), (
                "Invalid tile_dims, all tile_counts will be 1")

            active_dims = self.tile_dims <= self.extent

        if tile_counts is not None:
            try:
                i = int(tile_counts)
                self.tile_counts = (
                    i * np.ones(self.n_dims, dtype=np.dtype('i')))
            except:
                self.tile_counts = np.array(tile_counts, dtype=np.dtype('i'))

            assert self.tile_counts.ndim == 1
            assert self.tile_counts.size == self.n_dims
            assert all(self.tile_counts >= 1), "Detected non-pos tile_counts."
            assert any(self.tile_counts >= 2), "All tile_counts inactive."

            active_dims = self.tile_counts >= 2
            self.tile_dims = np.copy(self.extent)
            self.tile_dims[active_dims] = (
                self.extent / (self.tile_counts - 1))

        self.tiling_offsets = np.tile(self.origin, (self.n_tilings, 1))
        random_offsets = -np.random.random((self.n_tilings, len(active_dims)))
        self.tiling_offsets[:, active_dims] += (
            self.tile_dims[active_dims] * random_offsets)

        self.tiling_shape = np.ones(self.n_dims, dtype=np.dtype('i'))
        self.tiling_shape[active_dims] = (
            (np.ceil(self.extent / self.tile_dims) + 1)[active_dims])
        self.tiling_size = int(np.product(self.tiling_shape[active_dims]))
        self._n_features = n_tilings * self.tiling_size

        self.intercept = intercept
        if self.intercept:
            self._n_features += 1

    def as_vector(self, position):
        """
        Return a feature vector describing ``position`` using
        axis-aligned tile-coding.

        position: 1-D numpy array
            The position to be converted into a feature vector. Size
            must be equal to self.n_dims.
        """

        position = np.array(position, dtype=self.dtype)
        assert position.ndim == 1
        assert position.size == self.n_dims

        indices = position - self.tiling_offsets
        indices /= self.tile_dims
        indices = np.array(np.floor(indices), dtype=np.int16)
        lower_bound = np.all(indices >= 0, axis=1)
        upper_bound = np.all(indices < self.tiling_shape, axis=1)
        indices = np.hstack(
            (np.arange(self.n_tilings)[:, np.newaxis], indices))

        indices = indices[np.logical_and(lower_bound, upper_bound), :]

        vector = np.zeros([self.n_tilings] + list(self.tiling_shape))

        if indices.size != 0:
            vector[tuple(indices.T)] = 1.0

        vector = vector.flatten()
        if self.intercept:
            vector = np.concatenate((vector, [1]))

        return vector


class CircularCoarseCoding(FeatureExtractor):
    def __init__(
            self, n_circles, bounds, radius, arrangement='random',
            intercept=True):

        self.radius = radius
        self.bounds = bounds
        self.centres = np.random.random((n_circles, 2)) * bounds

        self._n_features = n_circles

        self.intercept = intercept
        if self.intercept:
            self._n_features += 1

    def as_vector(self, state):
        """
        State is convertible to a size-2 ndarray.
        """
        position = np.array(state)

        vector = (
            np.sqrt(np.sum((self.centres - position)**2, axis=1))
            < self.radius)

        if self.intercept:
            vector = np.concatenate((vector, [1]))

        return vector


class StateActionFeatureExtractor(FeatureExtractor):
    """
    Create a state-action feature extractor from a state feature extractor.

    Given a state-action pair (s, a), obtains features for the s from the
    provided state-feautre extractor, and then offsets those features in the
    returned vector according to a. Pads the rest with 0's.
    """

    def __init__(self, state_feature_extractor, n_actions):
        self.state_feature_extractor = state_feature_extractor
        self.n_actions = n_actions
        self.n_state_features = self.state_feature_extractor.n_features
        self._n_features = self.n_actions * self.n_state_features

    def as_vector(self, state, action):
        state_rep = self.state_feature_extractor.as_vector(state)

        vector = np.zeros(self.n_features)
        action = int(action)
        lo = action * self.n_state_features
        hi = lo + self.n_state_features
        vector[lo:hi] = state_rep

        return vector


def discounted_features(trajectory, feature_extractor, gamma):
    features = np.zeros(feature_extractor.n_features)

    for i, (s, a, r) in enumerate(trajectory):
        features += gamma**i * feature_extractor.as_vector(s, a)

    return features


if __name__ == "__main__":
    cc = CircularCoarseCoding(40, (1.0, 1.0), 0.3)
    vec = cc.as_vector((0.5, 0.5))
    print(vec)
