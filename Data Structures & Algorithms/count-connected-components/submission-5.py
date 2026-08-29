class Solution:
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        components = n
        component = {i: (lambda i=i: i) for i in list(range(n))}
        for e in edges:
            a, b = component[e[0]](), component[e[1]]()
            if component[a]() != component[b]():
                components -= 1
                if component[a]() < component[b]():
                    component[b] = lambda a=a: component[a]()
                else:
                    component[a] = lambda b=b: component[b]()
        return components