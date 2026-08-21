def rob1(nums: List[int]) -> int:
    nums = [0] + nums
    for i in range(2, len(nums)):
        nums[i] = max(nums[i] + nums[i-2], nums[i-1])
    return nums[-1]

class Solution:
    def rob(self, nums: List[int]) -> int:
        return max(rob1(nums[:-1]), rob1(nums[1:])) if len(nums) > 1 else nums[0]