def word_tuple(word: str) -> tuple(int):
    counts_list = [0 for i in range(26)]
    for letter in word:
        counts_list[ord(letter)-97] += 1
    return tuple(counts_list)

class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        d = {word_tuple(word): [] for word in strs}
        for word in strs:
            d[word_tuple(word)].append(word)
        return [d[group] for group in d]