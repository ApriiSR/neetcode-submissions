class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        big = cost[-1]
        mid = cost[-2]
        for i in range(len(cost)-3, -1, -1):
            lil = cost[i] + min(mid, big)
            big = mid
            mid = lil
        return min(mid, big)