class Solution:
    def getSum(self, a: int, b: int) -> int:
        c = a ^ b
        carry = 0
        d = 0
        for i in range(32):
            if ((c // 2**i) % 2) ^ carry:
                d |= 2**i
            if (((a // 2**i) % 2 or (b // 2**i) % 2) and carry) or (a // 2**i) % 2 and (b // 2**i) % 2:
                carry = 1
            else:
                carry = 0
        if d > 2**31:
            return -len(range(d, 2**32))
        else:
            return d