from collections import defaultdict
from collections.abc import Iterable, Iterator
import time
import numpy as np
import regex as re
import pickle


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens=None):
        "Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens."
        self.vocab = vocab
        self.merges = merges
        self.merge_lookup = {}
        for i, (left, right) in enumerate(merges):
            self.merge_lookup[(left, right)] = (i, left + right)
        self.special_tokens = special_tokens or []
        self.special_token_set = set(self.special_tokens)
        # Construct reverse bytes to int
        self.reverse_vocab = {}
        for k, v in vocab.items():
            self.reverse_vocab[v] = k
        self.base_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        if self.special_tokens:
            self.special_pat = re.compile(
                    "(" + "|".join(re.escape(tok) for tok in sorted(self.special_tokens, key=len, reverse=True)) + ")"
                )

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens=None):
        """Constructs and returns a Tokenizer from a serialized vocabulary and list of merges from the pkl format and a list of special tokens."""
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""

        special_set = set(self.special_tokens or [])

        # Split around special tokens first
        if self.special_tokens:
            parts = [p for p in self.special_pat.split(text) if p != ""]
        else:
            parts = [text]

        # Build tokens
        tokens = []
        for part in parts:
            if part in special_set:
                tokens.append(part)
            else:
                for pretok in self.base_pat.findall(part):
                    tokens.append([bytes([b]) for b in pretok.encode("utf-8")])

        # Apply merges to each normal token
        for idx, token in enumerate(tokens):
            if isinstance(token, str):  # special token, skip
                continue

            parts = token  # already a list of bytes from pre-tokenization
            while len(parts) > 1:
                best_rank = float('inf')
                best_idx = -1
                for i in range(len(parts) - 1):
                    pair = (parts[i], parts[i + 1])
                    if pair in self.merge_lookup:
                        rank = self.merge_lookup[pair][0]
                        if rank < best_rank:
                            best_rank = rank
                            best_idx = i
                if best_idx == -1:
                    break
                parts[best_idx] = parts[best_idx] + parts[best_idx + 1]
                parts.pop(best_idx + 1)
            tokens[idx] = parts

        # Convert to ids
        result = []
        for token in tokens:
            if isinstance(token, str):  # check if special token
                result.append(self.reverse_vocab[token.encode("utf-8")])
            else:
                result.extend(self.reverse_vocab[b] for b in token)

        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is required for memory-efficient tokenization of large files that we cannot directly load into memory."""
        for i in iterable:
            encoded = self.encode(i)
            yield from encoded

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        decoded = b""
        for i in ids:
            decoded += self.vocab[i]
        return decoded.decode("utf-8", errors="replace")


def get_compression_ratio():
    # sample 10 documents from TinyStories and OpenWebText
    tiny_stories_docs = []
    with open("data/raw_data/TinyStoriesV2-GPT4-valid.txt", "r") as f:
        curr_doc = ""
        for line in f:
            if "<|endoftext|>" in line:
                tiny_stories_docs.append(curr_doc)
                curr_doc = ""
                if len(tiny_stories_docs) >= 10:
                    break
            else:
                curr_doc += line
              
    owt_docs = []
    with open("data/raw_data/owt_valid.txt", "r") as f:
        curr_doc = ""
        for line in f:
            if "<|endoftext|>" in line:
                owt_docs.append(curr_doc)
                curr_doc = ""
                if len(owt_docs) >= 10:
                    break
            else:
                curr_doc += line
    
    tiny_stories_tokenizer = Tokenizer.from_files("output/output_tiny_bpe/vocab.pkl", "output/output_tiny_bpe/merges.pkl")
    tiny_stories_bytes = 0
    tiny_stories_tokens = 0
    for i in range(10):
        tiny_stories_bytes += len(tiny_stories_docs[i].encode('utf-8'))
        tiny_stories_tokens += len(tiny_stories_tokenizer.encode(tiny_stories_docs[i]))
    
    print(f"Tiny stories compression ratio: {tiny_stories_bytes / tiny_stories_tokens}")
    
    owt_tokenizer = Tokenizer.from_files("output/output_owt_bpe/vocab.pkl", "output/output_owt_bpe/merges.pkl")
    owt_bytes = 0
    owt_tokens = 0
    
    start = time.time()
    for i in range(10):
        owt_bytes += len(owt_docs[i].encode('utf-8'))
        owt_tokenizer.encode(owt_docs[i])
    
    elapsed = time.time() - start
    
    # OWT compression ratio: {owt_bytes / owt_tokens},
    print(f"bytes / seconds: {owt_bytes / elapsed}")
    
    
def encode_dataset(tokenizer, input_path, output_path):
    with open(input_path, "r") as f, open(output_path, "wb") as out:
        chunk = []
        for token_id in tokenizer.encode_iterable(f):
            chunk.append(token_id)
            if len(chunk) >= 1000000:
                np.array(chunk, dtype=np.uint16).tofile(out)
                chunk = []
        if chunk:
            np.array(chunk, dtype=np.uint16).tofile(out)
            
def main():
    tiny_tokenizer = Tokenizer.from_files("output/output_tiny_bpe/vocab.pkl", "output/output_tiny_bpe/merges.pkl", ["<|endoftext|>"])
    owt_tokenizer = Tokenizer.from_files("output/output_owt_bpe/vocab.pkl", "output/output_owt_bpe/merges.pkl", ["<|endoftext|>"])

    encode_dataset(tiny_tokenizer, "data/raw_data/TinyStoriesV2-GPT4-train.txt", "output/tiny_train.bin")
    encode_dataset(tiny_tokenizer, "data/raw_data/TinyStoriesV2-GPT4-valid.txt", "output/tiny_valid.bin")
    encode_dataset(owt_tokenizer, "data/raw_data/owt_train.txt", "output/owt_train.bin")
    encode_dataset(owt_tokenizer, "data/raw_data/owt_valid.txt", "output/owt_valid.bin")
            
        
if __name__ == "__main__":
    main()