class Solution:
    def carFleet(self, target: int, position: List[int], speed: List[int]) -> int:
        result = 1
        cars = [None] * target
        for i in range(len(position)):
            cars[position[i]] = (position[i], speed[i])
        cars = [car for car in cars if car is not None]
        lead = cars.pop()
        while cars:
            if (target-lead[0])/lead[1] >= (target-cars[-1][0])/cars[-1][1]:
                cars.pop()
            else:
                result += 1
                lead = cars.pop()
        return result