from collections import defaultdict
from typing import List, Dict, Tuple
import os

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

    def tokenize_file(self, input_file: str, output_file: str) -> Dict[str, float]:
        with open(input_file, 'r') as f:
            text = f.read()

        words = text.split()
        tokens = self.tokenize(text)

        with open(output_file, 'w') as f:
            f.write(' '.join(tokens))

        # Compute statistics
        stats = {
            "total_words": len(words),
            "total_tokens": len(tokens),
            "token_word_ratio": len(tokens) / len(words),
            "unique_tokens": len(set(tokens)),
            "vocab_usage": len(set(tokens)) / len(self.vocab),
        }

        return stats

def main():
    # Training
    corpus = ["the", "highest", "higher", "lower", "lowest", "cooler", "coolest", "the quick brown fox jumps over the lazy dog"]
    tokenizer = BPETokenizer(vocab_size=100)
    tokenizer.train(corpus)

    print("Vocabulary:", tokenizer.vocab)
    print("Merges:", tokenizer.merges)

    # Tokenize a file
    input_file = "input.txt"
    output_file = "/Users/siyanda/Desktop/CSC3003S/output_tokenized.txt"

    # Create a sample input file
    with open(input_file, 'w') as f:
        f.write("the quick brown fox jumps over the lazy dog")

    # Tokenize the file and get statistics
    stats = tokenizer.tokenize_file(input_file, output_file)

    print("\nTokenization Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")

    print(f"\nTokenized output has been written to {output_file}")

    # Read and print the tokenized output
    with open(output_file, 'r') as f:
        tokenized_text = f.read()
    print(f"\nTokenized text: {tokenized_text}")

    # Cleanup
    os.remove(input_file)
    os.remove(output_file)

if __name__ == "__main__":
    main()