class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        n1 = sum((ord(num1[-1-i]) - 48) * 10**i for i in range(len(num1)))
        n2 = sum((ord(num2[-1-i]) - 48) * 10**i for i in range(len(num2)))
        return str(n1*n2)