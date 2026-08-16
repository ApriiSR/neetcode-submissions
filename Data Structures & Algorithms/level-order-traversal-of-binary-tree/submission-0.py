# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if root:
            left = self.levelOrder(root.left)
            right = self.levelOrder(root.right)
            if len(left) < len(right):
                return [[root.val]] + [left[i] + right[i] for i in range(len(left))] + right[len(left):]
            else:
                return [[root.val]] + [left[i] + right[i] for i in range(len(right))] + left[len(right):]
        return []