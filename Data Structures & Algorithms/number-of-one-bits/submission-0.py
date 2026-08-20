class Solution:
    def hammingWeight(self, n: int) -> int:
        a = 0
        while n:
            a += n % 2
            n = n // 2
        return a