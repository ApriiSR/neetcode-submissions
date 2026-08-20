class Solution:
    def reverse(self, x: int) -> int:
        x = str(x)
        if x[0] == "-":
            x = x[-1:0:-1]
            if len(x) > 10 or (len(x) == 10 and x > "2147483648"):
                return 0
            else:
                return int("-" + x)
        else:
            x = x[::-1]
            if len(x) > 10 or (len(x) == 10 and x > "2147483647"):
                return 0
            else:
                return int(x)