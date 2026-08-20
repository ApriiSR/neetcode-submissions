import numpy

class Solution:
    def getSum(self, a: int, b: int) -> int:
        if a < 0:
            a = 2**32 + a
        if b < 0:
            b = 2**32 + b
        a = f"{a:032b}"
        b = f"{b:032b}"
        c = [0]*32
        carry = 0
        for i in range(31, -1, -1):
            c[i] = str(int(a[i]) ^ int(b[i]) ^ carry)
            if sum([int(a[i]),int(b[i]),carry]) >= 2:
                carry = 1
            else:
                carry = 0
        d = int("".join(c), 2)
        if d >= 2**31:
            return d - 2**32
        else:
            return d