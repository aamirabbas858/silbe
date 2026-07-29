"""
Stage 1 — turn German Wikipedia into one clean text file.

Wikipedia publishes its dumps as XML full of templates, infoboxes, tables and
reference markers. Stripping that reliably is fiddly work, and doing it badly
leaves debris the model happily learns to reproduce.

Wikimedia also publishes an already-extracted plain-text version on Hugging
Face, so we use that instead. It is the same content with the markup already
removed by the people who own the format.

Nothing here understands German. This stage only decides which articles are
worth keeping and writes them out in a consistent shape.

    python data/prepare.py --target-mb 500

Output: data/corpus.txt, and a small summary of what went in.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq
import requests
from tqdm import tqdm

# Twenty shards of German Wikipedia, already extracted to plain text.
# We pull them in order and stop as soon as we have enough.
REPO = "wikimedia/wikipedia"
CONFIG = "20231101.de"
SHARD_URL = (
    f"https://huggingface.co/datasets/{REPO}/resolve/main/{CONFIG}/"
    "train-{i:05d}-of-00020.parquet"
)
N_SHARDS = 20

# Articles shorter than this are stubs, disambiguation pages or redirects.
# They are mostly title and boilerplate, and teach sentence structure poorly.
MIN_CHARS = 600

# An article that is nearly all short lines is a list — filmographies, squad
# lists, discographies. Real prose is what we are after.
MAX_SHORT_LINE_RATIO = 0.5
SHORT_LINE_CHARS = 40

DATA = Path(__file__).parent
RAW = DATA / "raw"


def download_shard(i: int) -> Path:
    """Fetch one parquet shard, skipping it if we already have it."""
    RAW.mkdir(parents=True, exist_ok=True)
    dest = RAW / f"dewiki-{i:05d}.parquet"
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    url = SHARD_URL.format(i=i)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        # Written to a temp name first so an interrupted download is never
        # mistaken for a complete one on the next run.
        tmp = dest.with_suffix(".part")
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=f"shard {i}", leave=False
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
        tmp.rename(dest)
    return dest


def is_prose(text: str) -> bool:
    """True when the article reads like paragraphs rather than a list."""
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return False
    short = sum(1 for ln in lines if len(ln) < SHORT_LINE_CHARS)
    return short / len(lines) <= MAX_SHORT_LINE_RATIO


def clean(text: str) -> str:
    """Normalise whitespace. Deliberately light — the source is already text.

    The temptation is to strip more: numbers, parentheses, non-German
    characters. Resist it. Every rule removes real German along with the
    noise, and the model needs to see ordinary messy prose to write it.
    """
    text = text.replace("\r\n", "\n").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)          # runs of spaces
    text = re.sub(r"\n{3,}", "\n\n", text)       # runs of blank lines
    return text.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-mb", type=int, default=500,
                    help="stop once this much clean text has been written")
    ap.add_argument("--out", type=Path, default=DATA / "corpus.txt")
    ap.add_argument("--keep-shards", action="store_true",
                    help="keep the downloaded parquet files (≈780 MB each)")
    args = ap.parse_args()

    target = args.target_mb * 1024 * 1024
    written = kept = skipped_short = skipped_list = 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as out:
        for i in range(N_SHARDS):
            if written >= target:
                break

            shard = download_shard(i)
            pf = pq.ParquetFile(shard)

            # Read in row-group batches rather than loading a 780 MB shard
            # into memory at once — this has to run on 16 GB alongside
            # everything else the machine is doing.
            for batch in pf.iter_batches(batch_size=1000, columns=["title", "text"]):
                titles = batch.column("title").to_pylist()
                texts = batch.column("text").to_pylist()

                for title, text in zip(titles, texts):
                    if not text:
                        continue
                    text = clean(text)

                    if len(text) < MIN_CHARS:
                        skipped_short += 1
                        continue
                    if not is_prose(text):
                        skipped_list += 1
                        continue

                    # WikiText convention: the title as a heading, then the
                    # body. It gives the model a visible document boundary to
                    # learn, so it knows where an article starts and stops.
                    doc = f"= {title} =\n\n{text}\n\n"
                    out.write(doc)
                    written += len(doc.encode("utf-8"))
                    kept += 1

                if written >= target:
                    break

            if not args.keep_shards:
                shard.unlink(missing_ok=True)

            print(f"  shard {i}: {written / 1e6:.0f} MB written, {kept:,} articles")

    mb = args.out.stat().st_size / 1e6
    print(f"\n{args.out}  —  {mb:.0f} MB")
    print(f"  articles kept    {kept:,}")
    print(f"  skipped (short)  {skipped_short:,}")
    print(f"  skipped (list)   {skipped_list:,}")
    # A rough token count, useful for sizing training before the tokenizer
    # exists. German averages nearer 3 characters per token than English's 4.
    print(f"  ≈ {mb * 1e6 / 3.2 / 1e6:.0f}M tokens (rough estimate)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
