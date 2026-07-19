"""
Data loading for Fade, including fill-in-the-middle (FIM) support — a
training objective specific to code models (used by Codex, StarCoder,
DeepSeek-Coder, etc.) where the model learns to fill in a masked middle
section of a file given the prefix and suffix. This is what lets a code
model do things like autocomplete in the middle of a function, not just
append text at the end.

Still requires real tokenized code data on disk — see the note in the
original data.py stub. FIM only transforms existing token sequences; it
doesn't supply data by itself.
"""

import os
import random
import numpy as np
import torch

# special token IDs — reserve these at the end of your tokenizer's vocab
FIM_PREFIX = 64000 - 4
FIM_MIDDLE = 64000 - 3
FIM_SUFFIX = 64000 - 2
EOD = 64000 - 1


def apply_fim(tokens: np.ndarray, fim_rate: float) -> np.ndarray:
    """With probability fim_rate, rearrange a token sequence into
    [prefix] <FIM_SUFFIX> [suffix] <FIM_MIDDLE> [middle] <EOD> format,
    which teaches the model to condition on both sides of a gap."""
    if random.random() > fim_rate or len(tokens) < 8:
        return tokens

    n = len(tokens)
    a, b = sorted(random.sample(range(1, n), 2))
    prefix, middle, suffix = tokens[:a], tokens[a:b], tokens[b:]

    return np.concatenate([
        [FIM_PREFIX], prefix,
        [FIM_SUFFIX], suffix,
        [FIM_MIDDLE], middle,
        [EOD],
    ]).astype(tokens.dtype)


def get_dataloader(data_dir: str, batch_size: int, block_size: int,
                    split: str = "train", fim_rate: float = 0.5):
    bin_path = os.path.join(data_dir, f"{split}.bin")
    if not os.path.exists(bin_path):
        raise FileNotFoundError(
            f"No tokenized data found at {bin_path}. Tokenize your code corpus "
            f"first (see README) and write token ids here as a uint16/uint32 "
            f"binary file."
        )

    data = np.memmap(bin_path, dtype=np.uint32, mode="r")

    def generator():
        while True:
            xs, ys = [], []
            for _ in range(batch_size):
                i = random.randint(0, len(data) - block_size - 1)
                chunk = data[i:i + block_size + 1].astype(np.int64)
                if split == "train":
                    chunk = apply_fim(chunk, fim_rate)[: block_size + 1]
                    if len(chunk) < block_size + 1:
                        pad = np.full(block_size + 1 - len(chunk), EOD, dtype=np.int64)
                        chunk = np.concatenate([chunk, pad])
                xs.append(torch.from_numpy(chunk[:-1]))
                ys.append(torch.from_numpy(chunk[1:]))
            yield torch.stack(xs), torch.stack(ys)

    return generator()
