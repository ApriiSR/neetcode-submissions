class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n == 0:
            return 1
        elif n == 1:
            return x
        elif n > 1:
            a = x
            for _ in range(n.bit_length()-1):
                a *= a
            a *= self.myPow(x, n - (1 << (n.bit_length()-1)))
            return a
        else:
            return self.myPow(1/x, -n)