# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        node1 = l1
        node2 = l2
        carry = 0
        preroot = ListNode()
        prev = preroot
        while node1 or node2:
            val1 = node1.val if node1 else 0
            val2 = node2.val if node2 else 0
            total = val1 + val2 + carry
            prev.next = ListNode(total % 10)
            carry = total // 10
            prev = prev.next
            if node1: node1 = node1.next
            if node2: node2 = node2.next
        if carry: prev.next = ListNode(carry)
        return preroot.next
            