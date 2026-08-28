class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        best = nums[0]
        current = 0
        for num in nums:
            current = max(num, current + num)
            best = max(best, current)
        return best