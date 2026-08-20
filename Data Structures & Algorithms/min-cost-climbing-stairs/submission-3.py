class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        if len(cost) == 2:
            return min(cost[0], cost[1])
        big = cost[-1]
        mid = cost[-2]
        i = len(cost)-3
        while i >= 0:
            lil = cost[i] + min(mid, big)
            big = mid
            mid = lil
            i -= 1
        return min(lil, big)