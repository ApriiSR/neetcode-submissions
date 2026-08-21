class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        lands = sum(row.count("1") for row in grid)
        grid = [['0']*len(grid[0])] + grid + [['0']*len(grid[0])]
        for i in range(len(grid)):
            grid[i] = ['0'] + list(grid[i]) + ['0']
        islands = 0
        while True:
            r = 0
            while "1" not in grid[r]:
                r += 1
                if r == len(grid):
                    break
            if r == len(grid):
                break
            islands += 1
            c = grid[r].index("1")
            queue = [(r, c)]
            while queue:
                p = queue.pop()
                lands -= 1
                grid[p[0]][p[1]] = '0'
                for q in [(p[0]+1,p[1]), (p[0]-1,p[1]), (p[0],p[1]+1),(p[0],p[1]-1)]:
                    if q not in queue and grid[q[0]][q[1]] == '1':
                        queue.append(q)
        return islands
