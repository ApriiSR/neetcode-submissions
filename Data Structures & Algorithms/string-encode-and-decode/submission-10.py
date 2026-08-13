class Solution:

    def encode(self, strs: List[str]) -> str:
        if strs:
            return "paysutdlfmywvaugplk49w87ky9v478".join(strs) + "paysutdlfmywvaugplk49w87ky9v478"
        else:
            return ""

    def decode(self, s: str) -> List[str]:
        output = s.split("paysutdlfmywvaugplk49w87ky9v478")
        output.pop(-1)
        return output
