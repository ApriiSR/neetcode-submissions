class Solution:
    def findMin(self, nums: List[int]) -> int:
        start = 0
        end = len(nums)
        while end - start > 1:
            mid = start + (end-start)//2
            if nums[start] > nums[mid]:
                end = mid
            else:
                start = mid
        return nums[(start+1)%len(nums)]