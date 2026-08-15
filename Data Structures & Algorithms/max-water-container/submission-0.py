class Solution:
    def maxArea(self, heights: List[int]) -> int:
        best = 0
        for start in range(len(heights)-1):
            for end in range(start+1, len(heights)):
                best = max(best, (end-start)*min(heights[start],heights[end]))
        return best
