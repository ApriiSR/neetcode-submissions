# boxes are a type of line don't @ me
def isValidLine(line: list[str]) -> bool:
    for num in line:
        if num != "." and line.count(num) > 1:
            return False
    return True



class Solution:
    def box(self, a: int, board: List[List[str]]) -> List[str]:
        n = a // 3
        m = a % 3
        return board[3*n][3*m:3*m+3] + board[3*n+1][3*m:3*m+3] + board[3*n+2][3*m:3*m+3]

    def isValidSudoku(self, board: List[List[str]]) -> bool:
        valid = True
        # rows
        for row in board:
            valid *= isValidLine(row)
        cols = [[row[i] for row in board] for i in range(9)]
        for col in cols:
            valid *= isValidLine(col)
        boxes = [self.box(i, board) for i in range(9)]
        for box in boxes:
            valid *= isValidLine(box)
        return bool(valid)
