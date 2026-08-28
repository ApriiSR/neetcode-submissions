class Solution:
    def canJump(self, nums: List[int]) -> bool:
        if nums[0] >= len(nums) - 1:
            return True
        elif nums[0] == 0:
            return False
        else:
            return max(self.canJump(nums[i:]) for i in range(1, nums[0]+1))
