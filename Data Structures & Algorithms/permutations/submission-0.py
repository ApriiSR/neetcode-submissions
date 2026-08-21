def iterate(num: List[int], place = 0):
    if not num:
        return [-1]
    else:
        if num[-1] + 1 > place:
            return iterate(num[:-1], place + 1) + [0]
        else:
            num[-1] = num[-1] + 1
            return num

class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        perms = []
        num = [0] * len(nums)
        while -1 not in num:
            a = nums[:]
            perms.append([a.pop(d) for d in num])
            num = iterate(num)
        return perms