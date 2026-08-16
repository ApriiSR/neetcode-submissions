class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        max_length = 0
        length = 0
        for i in range(len(s)):
            if i+max_length >= len(s):
                break
            if length:
                length -= 1
            while i+length < len(s) and s[i+length] not in s[i:i+length]:
                length += 1
            max_length = max(max_length,length)
        return max_length