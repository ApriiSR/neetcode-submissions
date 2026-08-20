class strint:
    def __init__(self, x):
        if isinstance(x, int) and x < 0:
            self.sign = -1
            self.val = str(x)[1:]
        elif x:
            self.sign = 1
            self.val = str(x)
        else:
            self.sign = 1
            self.val = "0"
    
    def __gt__(self, other):
        if self.val == other.val:
            return False
        elif len(self.val) > len(other.val):
            return True
        elif len(other.val) > len(self.val):
            return False
        elif self.val[0] > other.val[0]:
            return True
        elif other.val[0] > self.val[0]:
            return False
        else:
            return strint(self.val[1:]) > strint(other.val[1:])
        
    def reverse(self):
        self.val = self.val[::-1]
        return self

class Solution:
    def reverse(self, x: int) -> int:
        x = strint(x).reverse()
        if x.sign == 1 and x > strint(2**31 - 1):
            return 0
        elif x.sign == -1 and x > strint(2**31):
            return 0
        else:
            return x.sign * int(x.val)
        