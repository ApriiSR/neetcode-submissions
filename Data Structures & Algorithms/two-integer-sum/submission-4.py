class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        nums2 = sorted(nums)
        low = 0
        high = -1
        while nums2[low] + nums2[high] != target:
            if nums2[low] + nums2[high] < target:
                low += 1
            else:
                high -= 1
        if nums.index(nums2[low]) == nums.index(nums2[high]):
            x = nums.index(nums2[low])
            nums.pop(x)
            return [x, nums.index(nums2[low]) + 1]
        else:
            return sorted([nums.index(nums2[low]), nums.index(nums2[high])])