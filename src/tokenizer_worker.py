import regex as re
from collections import defaultdict

# GPT2 regex
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pre_tokenize_chunk(args):
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        counts = defaultdict(int)

        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-token

        # Remove special tokens before pre-tokenization
        cleaned_chunks = ""
        if special_tokens:
            cleaned_chunks = re.split("|".join([re.escape(token) for token in special_tokens]), chunk)

        # Use regex-based GPT-2 pre-tokenizer
        for cleaned_chunk in cleaned_chunks:
            for match in re.finditer(PAT, cleaned_chunk):
                indices = tuple(match.group().encode("utf-8"))
                # count adjacent pairs
                counts[indices] += 1

    return counts
