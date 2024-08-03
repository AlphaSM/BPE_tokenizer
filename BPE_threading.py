from collections import defaultdict
from typing import List, Dict, Tuple
import os
import threading
from concurrent.futures import ThreadPoolExecutor
import math

class BPETokenizer:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.vocab = ["<|endoftext|>"]
        self.merges = {}
        self.token_to_id = {"<|endoftext|>": 0}
        self.id_to_token = {0: "<|endoftext|>"}
        self.lock = threading.Lock()

    def train(self, corpus: List[str], batch_size: int = 1000, num_threads: int = 4):
        # Step 1: Pre-tokenization and word frequency computation (batched)
        word_freqs = self._compute_word_freqs_batched(corpus, batch_size, num_threads)

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
            self._add_token(new_token)

    def _compute_word_freqs_batched(self, corpus: List[str], batch_size: int, num_threads: int) -> Dict[str, int]:
        word_freqs = defaultdict(int)
        batches = [corpus[i:i + batch_size] for i in range(0, len(corpus), batch_size)]

        def process_batch(batch):
            local_freqs = defaultdict(int)
            for text in batch:
                words = text.split()
                for word in words:
                    local_freqs[word] += 1
            return local_freqs

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = executor.map(process_batch, batches)

        for local_freqs in results:
            for word, freq in local_freqs.items():
                word_freqs[word] += freq

        return word_freqs

    def _compute_alphabet(self, word_freqs: Dict[str, int]):
        alphabet = set()
        for word in word_freqs.keys():
            alphabet.update(word)
        for char in sorted(alphabet):
            self._add_token(char)

    def _add_token(self, token):
        with self.lock:
            if token not in self.token_to_id:
                id = len(self.token_to_id)
                self.token_to_id[token] = id
                self.id_to_token[id] = token

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

    def tokenize(self, text: str) -> List[int]:
        words = text.split()
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

        tokens = sum(splits, [])
        return [self.token_to_id[token] for token in tokens]

    def tokenize_file(self, input_file: str, output_file: str, batch_size: int = 1000, num_threads: int = 4) -> Dict[str, float]:
        with open(input_file, 'r') as f:
            text = f.read()

        words = text.split()
        token_ids = self.tokenize_batched(text, batch_size, num_threads)
        tokens = [self.id_to_token[id] for id in token_ids]

        with open(output_file, 'w') as f:
            f.write(' '.join(tokens))
            f.write('\n')
            f.write(' '.join(map(str, token_ids)))

        # Compute statistics
        stats = {
            "total_characters": len(text),
            "total_words": len(words),
            "total_tokens": len(tokens),
            "token_word_ratio": len(tokens) / len(words),
            "unique_tokens": len(set(tokens)),
            "vocab_usage": len(set(tokens)) / len(self.vocab),
        }

        return stats

    def tokenize_batched(self, text: str, batch_size: int, num_threads: int) -> List[int]:
        words = text.split()
        batches = [words[i:i + batch_size] for i in range(0, len(words), batch_size)]

        def process_batch(batch):
            return self.tokenize(' '.join(batch))

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(process_batch, batches))

        return [token for batch in results for token in batch]

def main():
    # Training
    corpus = ["the", "highest", "higher", "lower", "lowest", "cooler", "coolest", "the quick brown fox jumps over the lazy dog"]
    tokenizer = BPETokenizer(vocab_size=100)
    tokenizer.train(corpus, batch_size=2, num_threads=2)  # Using small batch size for this small example

    print("Vocabulary:", tokenizer.vocab)
    print("Merges:", tokenizer.merges)

    # Tokenize a file
    input_file = "input.txt"
    output_file = "/Users/siyanda/Desktop/CSC3003S/output_tokenized.txt"

    # Create a sample input file
    with open(input_file, 'w') as f:
        f.write("the cat is in the hat " * 1000)  # Repeat the text to create a larger input

    # Tokenize the file and get statistics
    stats = tokenizer.tokenize_file(input_file, output_file, batch_size=100, num_threads=4)

    print("\nTokenization Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")

    print(f"\nTokenized output has been written to {output_file}")

    # Read and print the first few tokens of the tokenized output
    with open(output_file, 'r') as f:
        tokenized_text = f.readlines()
    print(f"\nFirst 10 tokens: {' '.join(tokenized_text[0].split()[:10])}")
    print(f"First 10 token IDs: {' '.join(tokenized_text[1].split()[:10])}")

if __name__ == "__main__":
    main()