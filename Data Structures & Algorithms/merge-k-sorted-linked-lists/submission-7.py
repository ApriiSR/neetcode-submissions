# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:    
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        lists = [node for node in lists if node is not None]
        preroot = ListNode(NotImplemented)
        prev = preroot
        while lists:
            min_val = float('inf')
            for j in range(len(lists)):
                if lists[j].val == prev.val:
                    i = j
                    break
                else:
                    if lists[j].val < min_val:
                        i = j
                    min_val = min(min_val, lists[j].val)
            prev.next = lists[i]
            prev = prev.next
            lists[i] = lists[i].next
            if not lists[i]:
                del lists[i]
        return preroot.next