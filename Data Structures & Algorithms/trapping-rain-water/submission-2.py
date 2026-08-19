class Solution:
    def trap(self, height: List[int]) -> int:
        start = 0
        end = len(height)-1
        water = [0 for h in height]
        while start < end:
            for i in range(start,end+1):
                water[i] = max(water[i], min(height[start], height[end]))
            if height[start] < height[end]:
                start += 1
            else:
                end -= 1
        return sum(max(0, water[i] - height[i]) for i in range(len(height)))
