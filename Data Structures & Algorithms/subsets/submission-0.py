class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        result = []
        for i in range(2**len(nums)):
            subset = []
            bits = f"{i:0{len(nums)}b}"
            for j in range(len(nums)):
                if bits[j] == "1":
                    subset.append(nums[j])
            result.append(subset)
        return result