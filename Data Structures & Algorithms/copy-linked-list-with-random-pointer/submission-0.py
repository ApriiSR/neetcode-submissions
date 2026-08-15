"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""

class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        d = {}
        original = head
        while original:
            d[original] = Node(original.val)
            original = original.next
        d[None] = None
        original = head
        while original:
            d[original].next = d[original.next]
            d[original].random = d[original.random]
            original = original.next
        return d[head]