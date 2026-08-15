# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        size = 0
        node = head
        while node:
            size += 1
            node = node.next
        if size < 1:
            return None
        prehead = ListNode(next=head)
        prev_node = prehead
        node = head
        for i in range(size-n):
            prev_node = node
            node = node.next
        prev_node.next = node.next
        return prehead.next