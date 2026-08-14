# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        if list2 is None:
            return list1
        if list1 is None:
            return list2
        
        root = list1 if list1.val < list2.val else list2
        next1 = list1.next if list1.val < list2.val else list1
        next2 = list2 if list1.val < list2.val else list2.next

        current = root

        while next1 and next2:
            if next1.val < next2.val:
                current.next = next1
                current=next1
                next1=next1.next
            else:
                current.next=next2
                current=next2
                next2=next2.next

        if next1:
            current.next=next1
            return root
        else:
            current.next=next2
            return root
