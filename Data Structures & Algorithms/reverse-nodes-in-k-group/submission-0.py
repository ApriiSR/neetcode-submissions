# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        a = []
        node = head
        while node:
            a.append(node.val)
            node = node.next
        groups = len(a)//k
        a = sum([a[i*k:(i+1)*k][::-1] for i in range(groups)], []) + a[groups*k:]
        dummy = ListNode()
        node = dummy
        for i in range(0,len(a)):
            node.next = ListNode(a[i])
            node = node.next
        return dummy.next