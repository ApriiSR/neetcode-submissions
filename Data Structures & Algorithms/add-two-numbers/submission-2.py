# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

def decode(node):
    if node:
        return node.val + 10 * decode(node.next)
    else:
        return 0

def encode(n):
    if n:
        return ListNode(n % 10, encode(n//10))

class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        a = decode(l1)
        b = decode(l2)
        return encode(a+b) if a+b else ListNode(0)