"""
# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
"""

class Solution:
    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        if not node:
            return None
        nodes = {}
        queue = [node]
        while queue:
            u = queue.pop(0)
            nodes[u] = Node(u.val)
            for v in u.neighbors:
                if v not in nodes:
                    queue.append(v)
        for old, new in nodes.items():
            new.neighbors = [nodes[w] for w in old.neighbors]
        return nodes[node]