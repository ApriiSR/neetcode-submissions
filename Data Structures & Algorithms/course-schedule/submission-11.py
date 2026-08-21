class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        parent = {}
        child = {}
        for course in range(numCourses):
            child[course] = []
            parent[course] = set()
        for p in prerequisites:
            child[p[0]].append(p[1])
            parent[p[1]].add(p[0])
        starts = list(range(numCourses))
        for p in prerequisites:
            if p[1] in starts:
                starts.remove(p[1])
        while starts:
            u = starts.pop()
            queue = [u]
            while queue:
                v = queue.pop()
                if parent[v]:
                    continue
                queue += child[v]
                for w in child.pop(v):
                    parent[w].remove(v)
        return not child