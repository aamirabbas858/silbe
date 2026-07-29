"""
Stage 3 — turn the corpus into one flat array of token IDs.

Tokenising is slow; training reads the data thousands of times. Doing it once
up front turns a repeated cost into a fixed one, and leaves the trainer doing
nothing but memory-mapping an array and slicing it.

The output is raw uint16 with no structure — no records, no offsets, no
padding. Every training batch is a random slice of one long stream of German.
uint16 is valid for any vocabulary under 65,536 and halves both the file size
and the memory traffic during training, which matters on a machine sharing
16 GB with a browser.

    python data/encode.py --tokenizer tokenizer/silbe-16384.json

Output: data/train-{vocab}.bin, data/val-{vocab}.bin
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
DATA = Path(__file__).parent

# Read the corpus in chunks rather than whole. 531 MB of text becomes far more
# than 531 MB as Python strings and intermediate token lists.
CHUNK_CHARS = 8 * 1024 * 1024

# The last 1% is held out and never trained on. Its only job is to answer
# "is this model learning German, or memorising Wikipedia?" — a question the
# training loss cannot answer, because the training loss is the thing being
# optimised.
VAL_FRACTION = 0.01


def encode(corpus: Path, tok: Tokenizer, out_train: Path, out_val: Path) -> tuple[int, int]:
    total_bytes = corpus.stat().st_size
    val_start = int(total_bytes * (1 - VAL_FRACTION))

    n_train = n_val = 0
    read = 0

    with (
        corpus.open(encoding="utf-8") as f,
        out_train.open("wb") as ftrain,
        out_val.open("wb") as fval,
        tqdm(total=total_bytes, unit="B", unit_scale=True, desc="encoding") as bar,
    ):
        while True:
            chunk = f.read(CHUNK_CHARS)
            if not chunk:
                break

            ids = np.asarray(tok.encode(chunk).ids, dtype=np.uint16)

            # The split is by position in the file, not by shuffling. Articles
            # are contiguous here, so a random split would put the first half
            # of an article in train and the second half in validation — and
            # the model would score well on validation for the wrong reason.
            if read < val_start:
                ftrain.write(ids.tobytes())
                n_train += len(ids)
            else:
                fval.write(ids.tobytes())
                n_val += len(ids)

            n = len(chunk.encode("utf-8"))
            read += n
            bar.update(n)

    return n_train, n_val


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizer" / "silbe-16384.json")
    ap.add_argument("--corpus", type=Path, default=DATA / "corpus.txt")
    args = ap.parse_args()

    for p in (args.tokenizer, args.corpus):
        if not p.exists():
            print(f"missing: {p}")
            return 1

    tok = Tokenizer.from_file(str(args.tokenizer))
    vocab = tok.get_vocab_size()
    if vocab >= 65536:
        print(f"vocab {vocab} does not fit in uint16")
        return 1

    train_bin = DATA / f"train-{vocab}.bin"
    val_bin = DATA / f"val-{vocab}.bin"

    n_train, n_val = encode(args.corpus, tok, train_bin, val_bin)

    print(f"\ntrain  {n_train:>12,} tokens  →  {train_bin.name}")
    print(f"val    {n_val:>12,} tokens  →  {val_bin.name}")
    print(f"total  {n_train + n_val:>12,} tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
