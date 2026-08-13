class Solution:
    def letters_multiset(self, s: str) -> dict:
        multiset = {}
        for i in s:
            multiset[i] = multiset[i] + 1 if i in multiset else 1
        return multiset

    def isAnagram(self, s: str, t: str) -> bool:
        multiset_s = self.letters_multiset(s)
        multiset_t = self.letters_multiset(t)
        try:
            for i in multiset_s:
                if multiset_s[i] != multiset_t[i]:
                    return False
            for i in multiset_t:
                if multiset_s[i] != multiset_t[i]:
                    return False
        except:
            return False
        return True
    