class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        DYNAMIC_PROGRAMMING = [[None for _ in range(m)] for _ in range(n)]
        for i in range(n):
            for j in range(m):
                if 0 in [i, j]:
                    DYNAMIC_PROGRAMMING[i][j] = 1
                else:
                    DYNAMIC_PROGRAMMING[i][j] = DYNAMIC_PROGRAMMING[i-1][j] + DYNAMIC_PROGRAMMING[i][j-1]
        return DYNAMIC_PROGRAMMING[n-1][m-1]