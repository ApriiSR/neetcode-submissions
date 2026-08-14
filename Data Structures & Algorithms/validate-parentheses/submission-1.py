class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        for bracket in s:
            if bracket in ["(", "{", "["]:
                stack.append(bracket)
            else:
                if not stack:
                    return False
                elif bracket == ")" and stack.pop() != "(":
                    return False
                elif bracket == "}" and stack.pop() != "{":
                    return False
                elif bracket == "]" and stack.pop() != "[":
                    return False
        return not stack
                    