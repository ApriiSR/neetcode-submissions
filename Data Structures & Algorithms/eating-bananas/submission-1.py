import math

def canEat(piles, h, k):
    if k == 0:
        return False
    else:
        return sum([math.ceil(pile/k) for pile in piles]) <= h

class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        start = 0
        end = max(piles) + 1
        while end > start:
            print(start, end)
            mid = start + (end - start) // 2
            if canEat(piles, h, mid):
                end = mid
            else:
                start = mid + 1
        return start