# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def reorderList(self, head: Optional[ListNode]) -> None:
        if not head:
            return head
        nodes = [head]
        while nodes[-1].next:
            nodes.append(nodes[-1].next)
        a = 0
        b = len(nodes)-1
        while a < b:
            nodes[a].next = nodes[b]
            if a+1 < b:
                nodes[b].next = nodes[a+1]
                a += 1
                b -= 1
            else:
                nodes[b].next = None
                break
        if a == b:
            nodes[a].next = None
