# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    @property
    def children(self):
        return [self.left, self.right]
    def __iter__(self):
        return [].__iter__()
    def __next__(self):
        return None

class set2:
    def __init__(self):
        self.items = []
        self._seen = set()
    def add(self, xs):
        for x in xs:
            if x is not None and x not in self._seen:
                self._seen.add(x)
                self.items.append(x)
    def pop(self, n):
        return self.items.pop(n)
    def __iter__(self):
        return iter(self.items)
    def __bool__(self):
        return bool(self.items)


class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        current_frontier = [root]
        result = []
        while current_frontier:
            result.append([node.val for node in current_frontier])
            next_frontier = set2()
            while current_frontier:
                node = current_frontier.pop(0)
                if node:
                    next_frontier.add(node.children)
            current_frontier = next_frontier
        return result