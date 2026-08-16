# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if root:
            return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1
        else:
            return 0

    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        if root:
            return self.isBalanced(root.left) and self.isBalanced(root.right) and abs(self.maxDepth(root.left) - self.maxDepth(root.right)) < 2
        else:
            return True