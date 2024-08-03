import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import math
import time
import re

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.token_id = None

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str, token_id: int):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
        node.token_id = token_id

    def search_longest_prefix(self, word: str) -> Tuple[str, int]:
        node = self.root
        prefix = ""
        last_match = ""
        last_token_id = None

        for char in word:
            if char in node.children:
                prefix += char
                node = node.children[char]
                if node.is_end:
                    last_match = prefix
                    last_token_id = node.token_id
            else:
                break

        return last_match, last_token_id

def process_batch(batch):
    local_freqs = defaultdict(int)
    for line in batch:
        words = re.findall(r'\S+|\s+', line)
        for word in words:
            if word.isspace():
                if word == ' ':
                    local_freqs['<|space|>'] += 1
                elif '\n' in word:
                    local_freqs['<|newline|>'] += len(word.split('\n')) - 1
                    local_freqs['<|space|>'] += len(word) - (len(word.split('\n')) - 1)
            else:
                local_freqs[word] += 1
    return local_freqs

class BPETokenizer:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.vocab = ["<|endoftext|>", "<|unknown|>", "<|space|>", "<|newline|>"]
        self.merges = {}
        self.token_to_id = {"<|endoftext|>": 0, "<|unknown|>": 1, "<|space|>": 2, "<|newline|>": 3}
        self.id_to_token = {0: "<|endoftext|>", 1: "<|unknown|>", 2: "<|space|>", 3: "<|newline|>"}
        self.trie = Trie()

    def train(self, corpus_file: str, batch_size: int = 1000, num_processes: int = 4, progress_callback=None):
        word_freqs = self._compute_word_freqs_batched(corpus_file, batch_size, num_processes)
        self._compute_alphabet(word_freqs)
        splits = {word: list(word) for word in word_freqs.keys()}

        total_iterations = self.vocab_size - len(self.vocab)
        for i in range(total_iterations):
            if progress_callback:
                progress_callback(i, total_iterations, prefix='Training Progress:', suffix='Complete', length=50)

            pair_freqs = self._compute_pair_freqs(splits, word_freqs)
            if not pair_freqs:
                break

            best_pair = max(pair_freqs, key=pair_freqs.get)
            self._merge_pair(best_pair, splits)
            new_token = ''.join(best_pair)
            self.merges[best_pair] = new_token
            self.vocab.append(new_token)
            self._add_token(new_token)

            # Implement early stopping
            if len(self.vocab) % 1000 == 0:
                compression_rate = self._calculate_compression_rate(word_freqs, splits)
                if compression_rate < 0.01:  # Stop if improvement is less than 1%
                    break

        # Build Trie after training
        self._build_trie()

    def _compute_word_freqs_batched(self, corpus_file: str, batch_size: int, num_processes: int) -> Dict[str, int]:
        word_freqs = defaultdict(int)

        with open(corpus_file, 'r', encoding='utf-8') as f:
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                batch = []
                for line in f:
                    batch.append(line)
                    if len(batch) == batch_size:
                        future = executor.submit(process_batch, batch)
                        for word, freq in future.result().items():
                            word_freqs[word] += freq
                        batch = []

                if batch:
                    future = executor.submit(process_batch, batch)
                    for word, freq in future.result().items():
                        word_freqs[word] += freq

        return word_freqs

    def _compute_alphabet(self, word_freqs: Dict[str, int]):
        alphabet = set()
        for word in word_freqs.keys():
            if word not in ['<|space|>', '<|newline|>']:
                alphabet.update(word)
        for char in sorted(alphabet):
            self._add_token(char)

    def _add_token(self, token):
        if token not in self.token_to_id:
            id = len(self.token_to_id)
            self.token_to_id[token] = id
            self.id_to_token[id] = token

    def _compute_pair_freqs(self, splits: Dict[str, List[str]], word_freqs: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            if word not in ['<|space|>', '<|newline|>']:
                split = splits[word]
                if len(split) == 1:
                    continue
                split_ids = np.array([self.token_to_id[token] for token in split])
                pairs = np.vstack((split_ids[:-1], split_ids[1:])).T
                unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
                for pair, count in zip(unique_pairs, counts):
                    pair_freqs[tuple(self.id_to_token[id] for id in pair)] += count * freq
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

    def _build_trie(self):
        for token, token_id in self.token_to_id.items():
            self.trie.insert(token, token_id)

    def _calculate_compression_rate(self, word_freqs: Dict[str, int], splits: Dict[str, List[str]]) -> float:
        original_chars = sum(len(word) * freq for word, freq in word_freqs.items())
        compressed_chars = sum(len(splits[word]) * freq for word, freq in word_freqs.items())
        return (original_chars - compressed_chars) / original_chars

    def tokenize(self, text: str) -> List[int]:
        tokens = []
        for item in re.findall(r'\S+|\s+', text):
            if item.isspace():
                if item == ' ':
                    tokens.append(self.token_to_id['<|space|>'])
                elif '\n' in item:
                    for char in item:
                        if char == '\n':
                            tokens.append(self.token_to_id['<|newline|>'])
                        else:
                            tokens.append(self.token_to_id['<|space|>'])
            else:
                word_tokens = self._tokenize_word(item)
                tokens.extend(word_tokens)
        return tokens

    def _tokenize_word(self, word: str) -> List[int]:
        tokens = []
        while word:
            prefix, token_id = self.trie.search_longest_prefix(word)
            if prefix:
                tokens.append(token_id)
                word = word[len(prefix):]
            else:
                tokens.append(self.token_to_id['<|unknown|>'])
                word = word[1:]
        return tokens

    def tokenize_file(self, input_file: str, output_file: str, batch_size: int = 1000, num_processes: int = 4) -> Dict[str, float]:
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()

        token_ids = self.tokenize_batched(text, batch_size, num_processes)
        tokens = [self.id_to_token[id] for id in token_ids]

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(' '.join(tokens))
            f.write('\n')
            f.write(' '.join(map(str, token_ids)))

        stats = {
            "total_characters": len(text),
            "total_tokens": len(tokens),
            "unique_tokens": len(set(tokens)),
            "vocab_usage": len(set(tokens)) / len(self.vocab),
        }

        return stats

    def tokenize_batched(self, text: str, batch_size: int, num_processes: int) -> List[int]:
        batches = [text[i:i + batch_size] for i in range(0, len(text), batch_size)]

        def process_batch(batch):
            return self.tokenize(batch)

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            results = list(executor.map(process_batch, batches))

        return [token for batch in results for token in batch]

def print_progress(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', print_end="\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    if iteration == total: 
        print()

def main():
    corpus_file = "corpus.txt"

    # Check if corpus file exists and is not empty
    if not os.path.exists(corpus_file):
        print(f"Error: Corpus file '{corpus_file}' not found.")
        return
    if os.path.getsize(corpus_file) == 0:
        print(f"Error: Corpus file '{corpus_file}' is empty.")
        return

    print(f"Corpus file '{corpus_file}' found and is not empty.")

    start = time.time()
    tokenizer = BPETokenizer(vocab_size=10000)
    end = time.time()
    print(f"Initialization time: {end - start:.2f} seconds")

    print("Starting training process...")
    start1 = time.time()
    
    tokenizer.train(corpus_file, batch_size=1000, num_processes=multiprocessing.cpu_count(), progress_callback=print_progress)
    
    end1 = time.time()
    print(f"Training time: {end1 - start1:.2f} seconds")

    print("Vocabulary size:", len(tokenizer.vocab))
    print("First 10 vocabulary items:", tokenizer.vocab[:10])
    print("Number of merges:", len(tokenizer.merges))

    input_file = "input.txt"
    output_file = "output_tokenized.txt"

    # Check if input file exists and is not empty
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return
    if os.path.getsize(input_file) == 0:
        print(f"Error: Input file '{input_file}' is empty.")
        return

    print(f"Input file '{input_file}' found and is not empty.")

    print("Starting tokenization process...")
    start2 = time.time()
    stats = tokenizer.tokenize_file(input_file, output_file, batch_size=1000, num_processes=multiprocessing.cpu_count())
    end2 = time.time()
    print(f"Tokenization time: {end2 - start2:.2f} seconds")

    print("\nTokenization Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")

    print(f"\nTokenized output has been written to {output_file}")

    with open(output_file, 'r', encoding='utf-8') as f:
        tokenized_text = f.readlines()
    print(f"\nFirst 10 tokens: {' '.join(tokenized_text[0].split()[:10])}")
    print(f"First 10 token IDs: {' '.join(tokenized_text[1].split()[:10])}")

if __name__ == "__main__":
    main()