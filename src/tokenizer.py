from collections import defaultdict
from collections.abc import Iterable, Iterator
import regex as re
import pickle


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens=None):
        "Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens."
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.special_token_set = set(self.special_tokens)
        # Construct reverse bytes to int
        self.reverse_vocab = {}
        for k, v in vocab.items():
            self.reverse_vocab[v] = k
        self.base_pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

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
            special_pat = re.compile(
                "(" + "|".join(re.escape(tok) for tok in sorted(self.special_tokens, key=len, reverse=True)) + ")"
            )
            parts = [p for p in special_pat.split(text) if p != ""]
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

        # Apply merges only to normal tokens
        for merge in self.merges:
            for idx, token in enumerate(tokens):
                if isinstance(token, str):  # special token
                    continue
                if len(token) <= 1:
                    continue

                i = 0
                new_token = []
                while i < len(token):
                    if i + 1 < len(token) and token[i] == merge[0] and token[i + 1] == merge[1]:
                        new_token.append(token[i] + token[i + 1])
                        i += 2
                    else:
                        new_token.append(token[i])
                        i += 1
                tokens[idx] = new_token

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
