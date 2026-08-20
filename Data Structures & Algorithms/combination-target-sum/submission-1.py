class Solution:
    def combinationSum(self, nums: List[int], target: int) -> List[List[int]]:
        if target <= 0:
            return []
        output = []
        if target in nums:
            output.append([target])
        for i in range(len(nums)):
            for x in self.combinationSum(nums[i:], target-nums[i]):
                output.append([nums[i]] + x)
        return output