import numpy as np

class Solution:
    def climbStairs(self, n: int) -> int:
        return int(np.linalg.matrix_power(np.array([[1, 1], [1, 0]]), n)[0, 0])