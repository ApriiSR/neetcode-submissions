import heapq

class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        a = [-1000] * k
        for num in nums:
            heapq.heappush(a, num)
            heapq.heappop(a)
        return a[0]