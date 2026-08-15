class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        result = []
        nums = sorted(nums)
        for low in range(len(nums)-2):
            mid = low + 1
            high = len(nums) - 1
            while mid < high:
                if nums[mid] + nums[high] == -nums[low]:
                    if (nums[low], nums[mid], nums[high]) not in result:
                        result.append((nums[low], nums[mid], nums[high]))
                    mid += 1
                    high -= 1
                elif nums[mid] + nums[high] < -nums[low]:
                    mid += 1
                else:
                    high -= 1
        return result
