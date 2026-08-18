# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def goodNodes(self, root: TreeNode, greatest = float('-inf')) -> int:
        if not root:
            return 0
        else:
            a = max(root.val, greatest)
            return (root.val >= greatest) + self.goodNodes(root.left, a) + self.goodNodes(root.right, a)