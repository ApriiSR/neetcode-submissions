class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        row_length = len(matrix[0])
        def nums(n: int) -> int:
            return matrix[n // row_length][n % row_length]
        start = 0
        end = row_length * len(matrix)
        while end - start > 1:
            mid = start + (end-start) // 2
            if nums(mid) < target:
                start = mid + 1
            elif nums(mid) > target:
                end = mid
            else:
                return True
        return not (start == end or nums(start) != target)