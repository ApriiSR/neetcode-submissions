import math

class Solution:
    def carFleet(self, target: int, position: List[int], speed: List[int]) -> int:
        result = 1
        cars = sorted([(position[i], speed[i]) for i in range(len(position))])
        lead = cars.pop()
        while cars:
            print(f"{math.ceil((target-lead[0])/lead[1])} >= {math.ceil((target-cars[-1][0])/cars[-1][1])}")
            if (target-lead[0])/lead[1] >= (target-cars[-1][0])/cars[-1][1]:
                cars.pop()
            else:
                result += 1
                lead = cars.pop()
        return result