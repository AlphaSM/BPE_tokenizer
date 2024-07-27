from collections import defaultdict
import re

class BPETokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size  # Vocabulary size
        self.vocab = {}  # Vocabulary dictionary
        self.merges = {}  # Merges dictionary
        self.word_freqs = defaultdict(int)  # Word frequencies

    def train(self, corpus):
        """Trains the tokenizer on a given text corpus."""
        # Preprocess the corpus: split into words and characters
        words = re.findall(r'\w+', corpus.lower())
        for word in words:
            self.word_freqs[' '.join(list(word)) + ' _'] += 1

        # Initialize vocabulary with characters
        chars = set(''.join(self.word_freqs.keys()))
        self.vocab = {c: i for i, c in enumerate(chars)}

        # Perform merges until the vocabulary size is reached
        while len(self.vocab) < self.vocab_size:
            pairs = self.get_stats()
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            self.merge_vocab(best_pair)
            print(f"Current vocab size: {len(self.vocab)}")  # Debugging line to monitor vocab size

    def get_stats(self):
        """Calculates the frequency of adjacent pairs in the current word frequencies."""
        pairs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def merge_vocab(self, pair):
        """Merges the most frequent pair into a new token."""
        new_token = ''.join(pair)
        self.merges[pair] = new_token
        self.vocab[new_token] = len(self.vocab)
        
        # Update word frequencies with the new token
        new_word_freqs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            new_word = re.sub(f'{pair[0]} {pair[1]}', new_token, word)
            new_word_freqs[new_word] += freq
        self.word_freqs = new_word_freqs

    def tokenize(self, text):
        """Tokenizes the input text into subword tokens based on the learned vocabulary."""
        words = re.findall(r'\w+', text.lower())
        tokens = []
        for word in words:
            word = ' '.join(list(word)) + ' _'
            while True:
                subwords = word.split()
                if len(subwords) == 1:
                    break
                for i in range(len(subwords) - 1):
                    if (subwords[i], subwords[i+1]) in self.merges:
                        subwords[i:i+2] = [self.merges[subwords[i], subwords[i+1]]]
                        break
                else:
                    break
                word = ' '.join(subwords)
            tokens.extend(subwords)
        return tokens

# Interactive part
def main():
    print("Welcome to the BPE Tokenizer!")
    
    while True:
        print("\n1. Train new tokenizer")
        print("2. Tokenize text")
        print("3. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            vocab_size = int(input("Enter desired vocabulary size: "))
            corpus = input("Enter or paste your training corpus: ")
            tokenizer = BPETokenizer(vocab_size)
            tokenizer.train(corpus)
            print(f"Tokenizer trained with vocabulary size: {len(tokenizer.vocab)}")

        elif choice == '2':
            if 'tokenizer' not in locals():
                print("Please train a tokenizer first.")
                continue
            text = input("Enter text to tokenize: ")
            tokens = tokenizer.tokenize(text)
            print("Tokenized text:", tokens)

        elif choice == '3':
            print("Thank you for using the BPE Tokenizer!")
            break

        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
