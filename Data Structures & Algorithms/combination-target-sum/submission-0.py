class Solution:
    def combinationSum(self, nums: List[int], target: int, needs_sort = True) -> List[List[int]]:
        if target <= 0:
            return []
        if needs_sort:
            nums = sorted(nums)
        output = []
        if target in nums:
            output.append([target])
        for i in range(len(nums)):
            for x in self.combinationSum(nums[i:], target-nums[i], False):
                output.append([nums[i]] + x)
        return output