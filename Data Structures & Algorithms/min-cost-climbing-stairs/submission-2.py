class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        a = cost[:]
        for i in range(len(cost)-3, -1, -1):
            a[i] = cost[i] + min(a[i+1], a[i+2])
        return min(a[0], a[1])