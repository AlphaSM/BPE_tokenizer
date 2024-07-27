from collections import defaultdict
from typing import List, Dict, Tuple

#BPE tokenizer from hugging face
class BPETokenizer:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.vocab = ["<|endoftext|>"]
        self.merges = {}

    def train(self, corpus: List[str]):
        # Step 1: Pre-tokenization and word frequency computation
        word_freqs = self._compute_word_freqs(corpus)

        # Step 2: Compute the base vocabulary (alphabet)
        self._compute_alphabet(word_freqs)

        # Step 3: Split words into individual characters
        splits = {word: list(word) for word in word_freqs.keys()}

        # Step 4: Learn merges until desired vocabulary size is reached
        while len(self.vocab) < self.vocab_size:
            pair_freqs = self._compute_pair_freqs(splits, word_freqs)
            if not pair_freqs:
                break

            best_pair = max(pair_freqs, key=pair_freqs.get)
            self._merge_pair(best_pair, splits)
            new_token = ''.join(best_pair)
            self.merges[best_pair] = new_token
            self.vocab.append(new_token)

    def _compute_word_freqs(self, corpus: List[str]) -> Dict[str, int]:
        word_freqs = defaultdict(int)
        for text in corpus:
            words = text.split()  # Simple split on whitespace for this example
            for word in words:
                word_freqs[word] += 1
        return word_freqs

    def _compute_alphabet(self, word_freqs: Dict[str, int]):
        alphabet = set()
        for word in word_freqs.keys():
            alphabet.update(word)
        self.vocab.extend(sorted(alphabet))

    def _compute_pair_freqs(self, splits: Dict[str, List[str]], word_freqs: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            split = splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs

    def _merge_pair(self, pair: Tuple[str, str], splits: Dict[str, List[str]]):
        for word in splits.keys():
            split = splits[word]
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [pair[0] + pair[1]] + split[i + 2:]
                else:
                    i += 1
            splits[word] = split

    def tokenize(self, text: str) -> List[str]:
        words = text.split()  # Simple split on whitespace for this example
        splits = [list(word) for word in words]
        
        for pair, merge in self.merges.items():
            for idx, split in enumerate(splits):
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merge] + split[i + 2:]
                    else:
                        i += 1
                splits[idx] = split

        return sum(splits, [])

# Example usage
if __name__ == "__main__":
    #doesnt merge 
    corpus = [ "A","B","C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P","Q","R", "S", "T", "U", "V", "W", "X", "Y", "Z",
               "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z" ]

    corpus = ["the", "highest", "higher", "lower", "lowest", "cooler", "coolest"]

    tokenizer = BPETokenizer(vocab_size=100)
    tokenizer.train(corpus)

    print("Vocabulary:", tokenizer.vocab)
    print("Merges:", tokenizer.merges)

    test_text = "the cat jumps up"
    tokens = tokenizer.tokenize(test_text)
    print("Tokenized:", tokens)