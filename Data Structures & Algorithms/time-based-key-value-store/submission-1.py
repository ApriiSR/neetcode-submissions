class TimeMap:

    def __init__(self):
        self.items = {}

    def set(self, key: str, value: str, timestamp: int) -> None:
        if key not in self.items:
            self.items[key] = [(timestamp, value)]
        else:
            self.items[key] += [(timestamp, value)]

    def get(self, key: str, timestamp: int) -> str:
        if key not in self.items:
            return ""
        start = 0
        end = len(self.items[key])
        while end - start > 1:
            mid = start + (end - start) // 2
            if self.items[key][mid][0] < timestamp:
                start = mid
            elif self.items[key][mid][0] > timestamp:
                end = mid
            else:
                return self.items[key][mid][1]
        return "" if start == end or self.items[key][start][0] > timestamp else self.items[key][start][1]