"""Train a BPE tokenizer."""

from collections import defaultdict
import time
import os
from typing import BinaryIO
import regex as re
from multiprocessing import Pool, cpu_count
import pickle
from pathlib import Path
import tracemalloc
import heapq

from src.tokenizer_worker import compile_re, pre_tokenize_chunk


class PairEntry:
    __slots__ = ("pair", "count", "vocab")

    def __init__(self, pair: tuple[int, int], count: int, vocab: dict[int, bytes]):
        self.pair = pair
        self.count = count
        self.vocab = vocab

    def __lt__(self, other: "PairEntry") -> bool:
        # Reverse comparison so it acts like max heap
        if self.count != other.count:
            return self.count > other.count

        self_key = (self.vocab[self.pair[0]], self.vocab[self.pair[1]])
        other_key = (other.vocab[other.pair[0]], other.vocab[other.pair[1]])
        return self_key > other_key


def build_pair_heap(
    counts: dict[tuple[int, int], int],
    vocab: dict[int, bytes],
) -> list[PairEntry]:
    heap = [PairEntry(pair, count, vocab) for pair, count in counts.items() if count > 0]
    heapq.heapify(heap)
    return heap


def pop_best_pair(
    heap: list[PairEntry],
    counts: dict[tuple[int, int], int],
) -> tuple[int, int] | None:
    while heap:
        entry = heapq.heappop(heap)
        current_count = counts.get(entry.pair, 0)

        if current_count <= 0:
            continue

        if current_count != entry.count:
            continue

        return entry.pair

    return None


def combine_counts(partial_counts):
    counts = defaultdict(int)
    for partial in partial_counts:
        for k, v in partial.items():
            counts[k] += v
    return counts


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    Adapted from cs336_basics.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def count_adjacent_pairs(
    indices: dict[bytes | tuple[int, ...], int],
) -> tuple[
    dict[tuple[int, int], int],
    dict[tuple[int, int], set[bytes | tuple[int, ...]]],
]:
    """
    Returns:
      - pair_counts: global count of each adjacent pair
      - pair_locations: which sequences currently contain each pair
    """
    pair_counts = defaultdict(int)
    pair_locations = defaultdict(set)

    for seq, freq in indices.items():
        seen = defaultdict(int)
        for a, b in zip(seq, seq[1:]):
            seen[(a, b)] += 1

        for pair, count in seen.items():
            pair_counts[pair] += count * freq
            pair_locations[pair].add(seq)

    return pair_counts, pair_locations


def merge_sequence(seq: bytes | tuple[int, ...], pair: tuple[int, int], new_index: int) -> tuple[int, ...]:
    """Merge one sequence only"""
    merged = []
    i = 0
    a, b = pair

    while i < len(seq):
        if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
            merged.append(new_index)
            i += 2
        else:
            merged.append(seq[i])
            i += 1

    return tuple(merged)


def merge(
    indices: dict[bytes | tuple[int, ...], int],
    pair_counts: dict[tuple[int, int], int],
    pair_locations: dict[tuple[int, int], set[bytes | tuple[int, ...]]],
    pair: tuple[int, int],
    new_index: int,
) -> set[tuple[int, int]]:
    """Update only sequences containing pair and return all pairs whose counts changed"""
    affected_seqs = list(pair_locations.get(pair, set()))
    affected_pairs = set()

    for old_seq in affected_seqs:
        old_freq = indices.pop(old_seq, 0)
        if old_freq == 0:
            continue

        old_local = defaultdict(int)
        for a, b in zip(old_seq, old_seq[1:]):
            old_local[(a, b)] += 1

        for old_pair, count in old_local.items():
            affected_pairs.add(old_pair)

            pair_counts[old_pair] -= count * old_freq
            if pair_counts[old_pair] <= 0:
                pair_counts.pop(old_pair, None)

            if old_pair in pair_locations:
                pair_locations[old_pair].discard(old_seq)
                if not pair_locations[old_pair]:
                    pair_locations.pop(old_pair, None)

        new_seq = merge_sequence(old_seq, pair, new_index)
        indices[new_seq] += old_freq

        new_local = defaultdict(int)
        for a, b in zip(new_seq, new_seq[1:]):
            new_local[(a, b)] += 1

        for new_pair, count in new_local.items():
            affected_pairs.add(new_pair)
            pair_counts[new_pair] += count * old_freq
            pair_locations[new_pair].add(new_seq)

    return affected_pairs


def pre_tokenize(input_path, num_processes, special_tokens):
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    jobs = [(input_path, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]

    with Pool(
        processes=num_processes,
        initializer=compile_re,
        initargs=(tuple(special_tokens),),
    ) as pool:
        partial_counts = pool.map(pre_tokenize_chunk, jobs)

    return combine_counts(partial_counts)


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    num_processes = max(1, cpu_count() - 1)

    indices = pre_tokenize(input_path, num_processes, special_tokens)
    merges = []
    # Initial vocab is 256 bytes + special tokens
    vocab = {x: bytes([x]) for x in range(256)}
    num_merges = vocab_size - len(vocab) - len(special_tokens)

    # Give special tokens an ID
    for i, special_token in enumerate(special_tokens):
        vocab[len(vocab)] = special_token.encode("utf-8")

    counts, pair_locations = count_adjacent_pairs(indices)
    heap = build_pair_heap(counts, vocab)

    # Find the most common pair
    for i in range(num_merges):
        if not counts:
            break

        pair = pop_best_pair(heap, counts)
        if pair is None:
            break

        # Merge that pair
        new_index = 256 + len(special_tokens) + i
        merges.append((vocab[pair[0]], vocab[pair[1]]))
        vocab[new_index] = vocab[pair[0]] + vocab[pair[1]]

        touched_pairs = merge(indices, counts, pair_locations, pair, new_index)

        for touched_pair in touched_pairs:
            count = counts.get(touched_pair, 0)
            if count > 0:
                heapq.heappush(heap, PairEntry(touched_pair, count, vocab))

        if len(heap) > 4 * max(1, len(counts)):
            heap = build_pair_heap(counts, vocab)

    return vocab, merges


def save_artifacts(vocab, merges, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open(out / "merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    with open(out / "vocab.txt", "w", encoding="utf-8") as f:
        for token_id, token_bytes in sorted(vocab.items()):
            f.write(f"{token_id}\t{token_bytes!r}\n")

    with open(out / "merges.txt", "w", encoding="utf-8") as f:
        for left, right in merges:
            f.write(f"{left!r}\t{right!r}\n")


def longest_token(vocab):
    token_id, token_bytes = max(vocab.items(), key=lambda kv: len(kv[1]))
    return token_id, token_bytes


def main():
    special_tokens = ["<|endoftext|>"]
    # tracemalloc.start(25)

    start = time.perf_counter()
    vocab, merges = train_bpe("data/raw_data/TinyStoriesV2-GPT4-train.txt", 10000, special_tokens)
    end = time.perf_counter()
    #current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    #snapshot = tracemalloc.take_snapshot()
    #tracemalloc.stop()

    save_artifacts(vocab, merges, "output_tiny_bpe")
    token_id, token_bytes = longest_token(vocab)
    print(f"Elapsed: {end - start}")
    print(f"Longest token: {token_id}, {token_bytes}")
    """print(f"Current traced memory: {current_bytes / (1024**2):.2f} MiB")
    print(f"Peak traced memory: {peak_bytes / (1024**2):.2f} MiB")
    print("\nTop 10 memory allocation sites:")
    for stat in snapshot.statistics("lineno")[:10]:
        print(stat)"""


if __name__ == "__main__":
    main()
