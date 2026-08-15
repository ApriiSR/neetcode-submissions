class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.stored = []
        self.cache = {}

    def get(self, key: int) -> int:
        if key in self.stored:
            self.stored.remove(key)
            self.stored.append(key)
        return self.cache[key] if key in self.stored else -1

    def put(self, key: int, value: int) -> None:
        self.cache[key] = value
        if key in self.stored:
            self.stored.remove(key)
        self.stored.append(key)
        if len(self.stored) > self.capacity:
            del self.stored[0]