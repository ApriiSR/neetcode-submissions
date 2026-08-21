class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        a = sorted([((v[0]**2 + v[1]**2)**0.5, v) for v in points])
        return [p[1] for p in a[0:k]]