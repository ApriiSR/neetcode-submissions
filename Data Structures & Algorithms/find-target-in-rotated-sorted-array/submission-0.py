def findMin(nums: List[int]) -> int:
    start = 0
    end = len(nums)
    while end - start > 1:
        mid = start + (end-start)//2
        if nums[start] > nums[mid]:
            end = mid
        else:
            start = mid
    return (start+1)%len(nums)

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        start = findMin(nums)
        end = start + len(nums)
        while end - start > 1:
            mid = start + (end-start) // 2
            if nums[mid % len(nums)] < target:
                start = mid + 1
            elif nums[mid % len(nums)] > target:
                end = mid
            else:
                return mid % len(nums)
        return -1 if start == end or nums[start % len(nums)] != target else start % len(nums)
        