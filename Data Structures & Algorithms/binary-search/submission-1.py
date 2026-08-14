def bin_search(nums: List[int], start: int, end: int, target: int):
    if end - start == 1:
        return start if nums[start] == target else -1
    elif end - start == 0:
        return -1
    else:
        mid = start + (end-start)//2
        if nums[mid] < target:
            return bin_search(nums, mid+1, end, target)
        elif nums[mid] > target:
            return bin_search(nums, start, mid, target)
        else:
            return mid

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        return bin_search(nums, 0, len(nums), target)
