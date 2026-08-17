class KthLargest:

    def __init__(self, k: int, nums: List[int]):
        self.stream = sorted(nums)
        self.k = k

    def add(self, val: int) -> int:
        i = 0
        while i < len(self.stream) and self.stream[i] < val:
            i += 1
        self.stream.insert(i, val)
        return self.stream[-self.k]
        
        
