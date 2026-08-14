class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        toOpen = {")": "(", "}": "{", "]": "["}
        for bracket in s:
            if bracket in "({[":
                stack.append(bracket)
            else:
                if not stack or stack.pop() != toOpen[bracket]:
                    return False
        return not stack