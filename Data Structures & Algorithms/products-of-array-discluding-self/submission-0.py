class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        a = nums.count(0)
        match a:
            case x if x > 1:
                return [0] * len(nums)
            case 1:
                zero = nums.index(0)
                prod = 1
                for i in nums[:zero]:
                    prod *= i
                for i in nums[zero+1:]:
                    prod *= i
                return [prod if num == 0 else 0 for num in nums]
            case 0:
                prod = 1
                for i in nums:
                    prod *= i
                return [prod // num for num in nums]