# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:    
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        preroot = ListNode(-10000)
        prev = preroot
        vals = [node.val for node in lists]
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
            if lists[i]:
                vals[i] = lists[i].val
            else:
                del lists[i]
                del vals[i]
        return preroot.next