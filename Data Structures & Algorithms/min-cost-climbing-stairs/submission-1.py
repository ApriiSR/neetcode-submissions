class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        a = [0] * len(cost)
        a[-1] = cost[-1]
        a[-2] = cost[-2]
        for i in range(len(cost)-3, -1, -1):
            a[i] = cost[i] + min(a[i+1], a[i+2])
        return min(a[0], a[1])