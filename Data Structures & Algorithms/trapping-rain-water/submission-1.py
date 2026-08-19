class Solution:
    def trap(self, height: List[int]) -> int:
        return sum(max(([0] + [min(max(height[0:i]),max(height[i+1:])) for i in range(1,len(height)-1)] + [0])[i]-height[i], 0) for i in range(len(height)))
