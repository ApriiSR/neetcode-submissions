class WordDictionary:

    def __init__(self):
        self.children = {}
        self.present = False

    def addWord(self, word: str) -> None:
        if word and not word[0] in self.children:
            self.children[word[0]] = WordDictionary()  
        if word:
            self.children[word[0]].addWord(word[1:])
        else:
            self.present = True

    def search(self, word: str) -> bool:
        if word:
            if word[0] == ".":
                return any(value.search(word[1:]) for value in self.children.values())
            elif word[0] in self.children:
                return self.children[word[0]].search(word[1:])
            else:
                return False
        else:
            return self.present