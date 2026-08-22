"""
Definition of Interval:
class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
"""

import heapq

class Solution:
    def minMeetingRooms(self, intervals: List[Interval]) -> int:
        intervals.sort(key = lambda x: x.start)
        rooms_needed = 0
        intervals_active = []
        for i in intervals:
            heapq.heappush(intervals_active, i.end)
            while intervals_active[0] <= i.start:
                heapq.heappop(intervals_active)
            rooms_needed = max(rooms_needed, len(intervals_active))
        return rooms_needed
            