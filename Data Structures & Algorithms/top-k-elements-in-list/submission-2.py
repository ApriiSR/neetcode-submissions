class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        num_counts = {}
        for num in nums:
            num_counts[num] = num_counts[num] + 1 if num in num_counts else 1
        counts = sorted(list(num_counts.values()))[-k:]
        output = []
        for num in counts:
            output += [key for key, value in num_counts.items() if value == num]
        return list(set(output))