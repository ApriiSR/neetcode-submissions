class Solution:
    def hasDuplicate(self, nums: List[int]) -> bool:
        dict = {}
        for i in nums:
            dict[i] = dict[i] + 1 if i in dict else 1
        for i in nums:
            if dict[i] > 1:
                return True
        return False