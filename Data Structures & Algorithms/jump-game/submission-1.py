class Solution:
    def canJump(self, nums: List[int]) -> bool:
        dp = [None] * len(nums)
        dp[-1] = True
        for i in range(len(nums)-2, -1, -1):
            dp[i] = any(dp[j] for j in range(i+1, i+nums[i]+1))
        return dp[0]