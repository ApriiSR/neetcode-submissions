class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:
        while len(stones) > 1:
            stones = sorted(stones)
            if stones[-2] == stones[-1]:
                stones.pop() and stones.pop()
            else:
                stones [-2] = stones[-1] - stones[-2]
                stones.pop()
        if stones:
            return stones[0]
        else:
            return 0