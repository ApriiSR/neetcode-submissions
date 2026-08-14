class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        largest = max(heights)
        for i in range(largest+1):
            width = 0
            for j in heights:
                if j >= i:
                    width += 1
                else:
                    area = i * width
                    largest = max(largest, area)
                    width = 0
            area = i * width
            largest = max(largest, area)
        return largest
