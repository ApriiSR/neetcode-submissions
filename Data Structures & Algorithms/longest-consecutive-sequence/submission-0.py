class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        longest = 0
        numset = set(nums)
        while numset:
            current = 1
            start = numset.pop()
            lower = start - 1
            while lower in numset:
                current += 1
                numset.remove(lower)
                lower -= 1
            higher = start + 1
            while higher in numset:
                current += 1
                numset.remove(higher)
                higher += 1
            longest = max(longest, current)
        return longest