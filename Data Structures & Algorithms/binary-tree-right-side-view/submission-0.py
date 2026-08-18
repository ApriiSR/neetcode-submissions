# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

def levelOrder(root: Optional[TreeNode]) -> List[List[int]]:
    result = []
    if not root:
        return []
    current_frontier = deque([root])
    while current_frontier:
        result.append([node.val for node in current_frontier])
        next_frontier = deque()
        while current_frontier:
            node = current_frontier.popleft()
            if node.left:
                next_frontier.append(node.left)
            if node.right:
                next_frontier.append(node.right)
        current_frontier = next_frontier
    return result

class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        return [level[-1] for level in levelOrder(root)]