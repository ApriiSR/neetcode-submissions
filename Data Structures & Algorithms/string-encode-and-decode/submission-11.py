class Solution:

    def encode(self, strs: List[str]) -> str:
        return "".join([f"{len(s)}|{s}" for s in strs])

    def decode(self, s: str) -> List[str]:
        strs = []
        while s:
            a = s.index("|")
            strs.append(s[a+1:a+1+int(s[0:a])])
            s = s[a+1+int(s[0:a]):]
        return strs

