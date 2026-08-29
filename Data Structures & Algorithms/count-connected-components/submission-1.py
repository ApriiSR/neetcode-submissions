class Solution:
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        vertices = list(range(n))
        components = 0
        while vertices:
            components += 1
            queue = [vertices.pop()]
            while queue:
                v = queue.pop(0)
                for e in edges[::-1]:
                    if v in e:
                        edges.remove(e)
                        if e[0] in vertices:
                            queue.append(e[0])
                            vertices.remove(e[0])
                        if e[1] in vertices:
                            queue.append(e[1])
                            vertices.remove(e[1])
        return components