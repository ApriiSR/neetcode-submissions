class PrefixTree:

    def __init__(self):
        self.children = {}
        self.present = False

    def insert(self, word: str) -> None:
        if word and not word[0] in self.children:
            self.children[word[0]] = PrefixTree()  
        if word:
            self.children[word[0]].insert(word[1:])
        else:
            self.present = True

    def search(self, word: str) -> bool:
        if word:
            if word[0] in self.children:
                return self.children[word[0]].search(word[1:])
            else:
                return False
        else:
            return self.present


    def startsWith(self, prefix: str) -> bool:
        if prefix:
            if prefix[0] in self.children:
                return self.children[prefix[0]].startsWith(prefix[1:])
            else:
                return False
        else:
            return True
        