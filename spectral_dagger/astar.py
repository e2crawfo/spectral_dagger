import numpy as np
from heapq import heapify, heappush, heappop
from collections import defaultdict

from policy import Policy

one_norm_distance = lambda x, y: np.linalg.norm(x - y, ord='1')

class AStarPolicy(Policy):
    def __init__(self, start_location, goal_location, world):
        Astar(start_location, goal_location, heuristic, dist, neighbours)

    def reset(self, init_dist=None):
        pass

    def action_played(self, action):
        pass

    def observation_emitted(self, obs):
        pass

    def get_action(self):
        """
        Returns the action chosen by the policy, given the history of
        actions and observations it has encountered. Note that it should Note
        assume that the action that it returns actually gets played.
        """

        pass




def AStar(start, goal, heuristic, dist, neighbours):
    closed_set = set()

    came_from = {}
    g_score = defaultdict(lambda: np.inf)
    f_score = PriorityDict()

    g_score[start] = 0
    f_score[start] = g_score[start] + heuristic(start, goal)

    while f_score:
        current = f_score.smallest()

        if current == goal:
            return reconstruct_path(came_from, goal)

        closed_set.add(current)

        for neighbour in neighbours(current):
            if neighbour in closed_set:
                continue

            tentative_g_score = g_score[current] + dist(current, neighbour)

            if tentative_g_score < g_score[neighbour]:
                came_from[neighbour] = current
                g_score[neighbour] = tentative_g_score
                f_score[neighbour] = (
                    g_score[neighbour] + heuristic(neighbour, goal))


def reconstruct_path(came_from, current):
    total_path = [current]

    while current in came_from:
        current = came_from[current]
        total_path.append(current)

    return total_path


class PriorityDict(dict):
    """
    Retrieved from: http://code.activestate.com/recipes/
        522995-priority-dict-a-priority-queue-with-updatable-prio/

    Dictionary that can be used as a priority queue.

    Keys of the dictionary are items to be put into the queue, and values
    are their respective priorities. All dictionary methods work as expected.
    The advantage over a standard heapq-based priority queue is
    that priorities of items can be efficiently updated (amortized O(1))
    using code as 'thedict[item] = new_priority.'

    The 'smallest' method can be used to return the object with lowest
    priority, and 'pop_smallest' also removes it.

    The 'sorted_iter' method provides a destructive sorted iterator.
    """

    def __init__(self, *args, **kwargs):
        super(PriorityDict, self).__init__(*args, **kwargs)
        self._rebuild_heap()

    def _rebuild_heap(self):
        self._heap = [(v, k) for k, v in self.iteritems()]
        heapify(self._heap)

    def smallest(self):
        """Return the item with the lowest priority.

        Raises IndexError if the object is empty.
        """

        heap = self._heap
        v, k = heap[0]
        while k not in self or self[k] != v:
            heappop(heap)
            v, k = heap[0]
        return k

    def pop_smallest(self):
        """Return the item with the lowest priority and remove it.

        Raises IndexError if the object is empty.
        """

        heap = self._heap
        v, k = heappop(heap)
        while k not in self or self[k] != v:
            v, k = heappop(heap)
        del self[k]
        return k

    def __setitem__(self, key, val):
        # We are not going to remove the previous value from the heap,
        # since this would have a cost O(n).

        super(PriorityDict, self).__setitem__(key, val)

        if len(self._heap) < 2 * len(self):
            heappush(self._heap, (val, key))
        else:
            # When the heap grows larger than 2 * len(self), we rebuild it
            # from scratch to avoid wasting too much memory.
            self._rebuild_heap()

    def setdefault(self, key, val):
        if key not in self:
            self[key] = val
            return val
        return self[key]

    def update(self, *args, **kwargs):
        # Reimplementing dict.update is tricky -- see e.g.
        # http://mail.python.org/pipermail/python-ideas/2007-May/000744.html
        # We just rebuild the heap from scratch after passing to super.
        super(PriorityDict, self).update(*args, **kwargs)
        self._rebuild_heap()

    def sorted_iter(self):
        """Sorted iterator of the priority dictionary items.

        Beware: this will destroy elements as they are returned.
        """
        while self:
            yield self.pop_smallest()



if __name__ == "__main__":
    pass






