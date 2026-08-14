# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head is None:
            return head
        nodes = [head]
        while nodes[-1] is not None:
            nodes.append(nodes[-1].next)
        for i in range(len(nodes)-1):
            nodes[i].next = nodes[i-1]
        return nodes[-2]
