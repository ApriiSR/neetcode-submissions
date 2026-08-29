# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

def validate(root, min_val = float('-inf'), max_val = float('inf')):
    if not root:
        return True
    elif root.val <= min_val or root.val >= max_val:
        return False
    else:
        return validate(root.left, min_val, root.val) and validate(root.right, root.val, max_val)

class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        return validate(root)