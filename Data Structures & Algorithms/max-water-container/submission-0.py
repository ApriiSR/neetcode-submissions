class Solution:
    def maxArea(self, heights: List[int]) -> int:
        most = 0
        for start in range(len(heights)-1):
            for end in range(start+1, len(heights)):
                most = max(most, (end-start)*min(heights[start],heights[end]))
        return most
