import numpy as np

from spectral_dagger.utils.geometry import Position


class FeatureExtractor(object):
    def __init__(self):
        return NotImplementedError("Cannot instantiate FeatureExtractor.")

    @property
    def n_features(self):
        return self._n_features


class RectangularTiling(FeatureExtractor):
    def __init__(self, offset, bounds, granularity):
        self.offset = offset
        self.granularity = granularity

    def as_vector(self, state):
        try:
            position = Position(state)
        except:
            raise NotImplementedError()

        vector = np.array(
            [position in shape for shape in self.shapes], dtype=bool)

        return vector


class RectangularTileCoding(FeatureExtractor):
    def __init__(self, n_tilings, bounds, granularity, intercept=True):
        """
        Tile at the beginning and the end is the same.
        That is, there is wrap-around.

        bounds: 2-D vector giving the extent of the tiled region
        granularity: 2-D vector or a position
        offsets should be negative, and less than the granularity
        """

        self.granularity = np.array(granularity)
        self.bounds = np.array(bounds)
        self.n_tilings = n_tilings

        self.offsets = -np.random.random((n_tilings, 2))
        self.offsets *= self.granularity

        self.tiling_shape = np.ceil(self.bounds / self.granularity) + 1
        self.tiling_size = np.product(self.tiling_shape)
        self._n_features = n_tilings * self.tiling_size

        self.intercept = intercept
        if self.intercept:
            self._n_features += 1

    def as_vector(self, state):
        """
        State is convertible to a size-2 ndarray.
        """
        position = np.array(state)

        indices = position - self.offsets
        indices /= self.granularity
        indices = np.array(np.floor(indices), dtype=np.int16)

        def valid(x):
            v = x[0] >= 0 and x[0] < self.tiling_shape[0]
            v &= x[1] >= 0 and x[1] < self.tiling_shape[1]
            return v
        indices = np.array(filter(valid, indices))

        vector = np.zeros(
            (self.n_tilings, self.tiling_shape[0], self.tiling_shape[1]))

        if indices.size != 0:
            vector[
                np.arange(self.n_tilings),
                indices[:, 0], indices[:, 1]] = 1.0

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
    print vec
