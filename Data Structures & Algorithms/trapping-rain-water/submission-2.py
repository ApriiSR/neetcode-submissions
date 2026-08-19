class Solution:
    def trap(self, height: List[int]) -> int:
        best = 0
        start = 0
        end = len(height)-1
        water = [0 for h in height]
        while start < end:
            level = min(height[start], height[end])
            if level > best:
                for i in range(start,end+1):
                    water[i] = level
            best = max(best, level)
            if height[start] < height[end]:
                start += 1
            else:
                end -= 1
        return sum(max(0, water[i] - height[i]) for i in range(len(height)))