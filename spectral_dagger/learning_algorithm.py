
class LearningAlgorithm(object):
    """
    Learning algorithms learn from some of data (e.g. sampled trajectories
    or an explicit model such as an MDP), and yield a policy. The constructor
    should accept parameters of the learning algorithm. The fit function should
    take the data that the algorithm is to learn from and yield a policy.
    """

    def __init__(self):
        raise NotImplemented("Cannot instantiate LearningAlgorithm.")

    def fit(self, data):
        raise NotImplemented("Cannot instantiate LearningAlgorithm.")
