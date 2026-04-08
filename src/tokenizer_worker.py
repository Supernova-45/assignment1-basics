import regex as re
from collections import defaultdict

# GPT2 regex
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT_RE = re.compile(PAT)
SPLIT_RE = None


def compile_re(special_tokens: tuple[str, ...]):
    global SPLIT_RE
    if special_tokens:
        SPLIT_RE = re.compile("|".join(re.escape(token) for token in special_tokens))
    else:
        SPLIT_RE = None


def pre_tokenize_chunk(args) -> dict[bytes, int]:
    input_path, start, end = args
    with open(input_path, "rb") as f:
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        counts = defaultdict(int)

        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-token

        # Remove special tokens before pre-tokenization
        # Use regex-based GPT-2 pre-tokenizer to count adjacent pairs
        for cleaned_chunk in (SPLIT_RE.split(chunk) if SPLIT_RE else [chunk]):
            for match in PAT_RE.finditer(cleaned_chunk):
                counts[match.group().encode("utf-8")] += 1

    return dict(counts)
