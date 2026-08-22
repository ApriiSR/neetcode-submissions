import heapq

class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        a = []
        heapq.heapify(a)
        for num in nums:
            heapq.heappush(a, -num)
        for i in range(k-1):
            heapq.heappop(a)
        return -heapq.heappop(a)