# trying to implement the more correct estimator of hankel matrix
# entries, used by grinberg and precup condpsr

# basically, we have to through before hand, and get counts for all prefixes.
# only then can we go through and get the estimator for a trajectory,
# using all the nodes 

class TrajectoryTree(object):
    def __init__(self, val=None):
        self.count = 0
        self.val = val
        self.children = {}
        self.trajectory = None

    def get_count(self):
        return self.count

    def add_trajectory(self, trajectory):
        if not trajectory:
            return

        self.count += 1

        if not self.children and not trajectory:
            self.trajectory = trajectory

        elif not self.children:
            if self.trajectory[0] != trajectory[0]:

                first_item = trajectory[0]
                self.children[first_item] = TrajectoryTree(first_item)
                self.children[first_item].add_trajectory(trajectory[1:])

                first_item = self.trajectory[0]
                self.children[first_item] = TrajectoryTree(first_item)
                self.children[first_item].add_trajectory(self.trajectory[1:])

                self.trajectory = None
        else:
            first_item = trajectory[0]

            if first_item in self.children:
                self.children[first_item].add_trajectory(trajectory[1:])
            else:
                self.children[first_item] = TrajectoryTree(first_item)
                self.children[first_item].add_trajectory(trajectory[1:])

