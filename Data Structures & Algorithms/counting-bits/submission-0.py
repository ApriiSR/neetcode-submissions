def hammingWeight(n: int) -> int:
    a = 0
    while n:
        a += n % 2
        n = n // 2
    return a

class Solution:
    def countBits(self, n: int) -> List[int]:
        return [hammingWeight(i) for i in range(n+1)]