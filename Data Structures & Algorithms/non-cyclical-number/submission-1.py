class Solution:
    def isHappy(self, n: int) -> bool:
        visited = set()
        sq = {"0": 0, "1": 1, "2": 4, "3": 9, "4": 16, "5": 25, "6": 36, "7": 49, "8": 64, "9": 81}
        while n not in visited:
            visited.add(n)
            n = sum(sq[d] for d in str(n))
            if n == 1:
                return True
        return False
