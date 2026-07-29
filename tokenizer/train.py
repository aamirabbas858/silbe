"""
Stage 2 — learn how to cut German into pieces.

A model cannot read text. It reads integers. The tokenizer decides which
pieces of text get which integers, and that decision is more consequential in
German than in English.

English mostly builds meaning with separate words: "speed limit". German
welds them: "Geschwindigkeitsbegrenzung". A tokenizer has to cut somewhere,
and where it cuts determines how many tokens a sentence costs — which
determines how much German fits in the model's context window, and therefore
how much it can learn from each training step.

We use byte-level BPE (byte-pair encoding), the same family GPT-2 and Llama
use. It starts from raw bytes and repeatedly merges the most frequent adjacent
pair. Frequent words end up whole; rare ones stay in fragments. Because it
starts from bytes, no text is ever unrepresentable — umlauts, emoji and typos
all encode without an unknown token.

    python tokenizer/train.py --vocab-size 16384

Output: tokenizer/silbe-{vocab}.json, plus the measurements that make up the
tokenizer study.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tokenizers import Tokenizer, decoders, pre_tokenizers, processors
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

ROOT = Path(__file__).parent.parent
HERE = Path(__file__).parent

# Marks the boundary between articles. Giving the model an explicit token for
# "document ends here" is better than hoping it infers it from blank lines.
END_OF_TEXT = "<|endoftext|>"

# German compounds, ordered roughly by length. These are the measurement: how
# a tokenizer handles them is the whole question this project asks.
#
# The last one is real — it was an actual law until 2013, and it is the
# standard example of German's willingness to keep gluing.
COMPOUNDS = [
    "Hauptbahnhof",
    "Krankenversicherung",
    "Aufenthaltserlaubnis",
    "Wohnungsgeberbestätigung",
    "Geschwindigkeitsbegrenzung",
    "Bundesausbildungsförderungsgesetz",
    "Arbeitsunfähigkeitsbescheinigung",
    "Donaudampfschifffahrtsgesellschaft",
    "Rindfleischetikettierungsüberwachungsaufgabenübertragungsgesetz",
]

# A plain sentence, for measuring the ordinary case rather than the extreme.
SAMPLE = (
    "Der Hauptbahnhof in Berlin ist der größte Kreuzungsbahnhof Europas und "
    "wurde im Jahr 2006 nach elf Jahren Bauzeit eröffnet."
)


def train(corpus: Path, vocab_size: int, out: Path) -> Tokenizer:
    tok = Tokenizer(BPE(unk_token=None))

    # ByteLevel does two jobs. It maps every byte to a printable character, so
    # any input is representable; and add_prefix_space means "Bahnhof" at the
    # start of a line and " Bahnhof" mid-sentence become the same token rather
    # than two separate ones the model has to learn twice.
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=True)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[END_OF_TEXT],
        # A pair must appear at least twice before it earns a merge. Without
        # this, single typos in Wikipedia become permanent vocabulary entries.
        min_frequency=2,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    t0 = time.time()
    tok.train([str(corpus)], trainer)
    print(f"trained in {time.time() - t0:.0f}s")

    out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out))
    return tok


def measure(tok: Tokenizer, corpus: Path, sample_bytes: int = 5_000_000) -> dict:
    """The tokenizer study: four numbers that describe what this vocab costs."""
    text = corpus.open(encoding="utf-8").read(sample_bytes)

    ids = tok.encode(text).ids
    words = len(text.split())

    # Fertility — average tokens per whitespace-separated word. 1.0 would mean
    # every word is a single token. English sits near 1.3; German is worse,
    # and how much worse is what the vocabulary size controls.
    fertility = len(ids) / max(words, 1)

    # How the extreme cases split. This is the readable part of the result —
    # seeing a real word shatter into fragments makes the cost concrete in a
    # way a number does not.
    splits = {}
    for word in COMPOUNDS:
        ids = tok.encode(word).ids
        # Decode each token individually rather than reading .tokens. A
        # byte-level tokenizer stores 'ä' as its two UTF-8 bytes mapped to
        # printable stand-ins, so .tokens renders it as 'Ã¤'. Decoding turns
        # the bytes back into the character — which matters here, because the
        # whole point of this table is to be readable.
        splits[word] = [tok.decode([i]).strip() for i in ids]

    # Effective context — how much actual German fits in a 256-token window.
    # This is the number that matters for training: a worse tokenizer means
    # each step sees less language for the same compute.
    sent_ids = tok.encode(SAMPLE).ids
    words_per_256 = 256 * len(SAMPLE.split()) / max(len(sent_ids), 1)

    return {
        "vocab_size": tok.get_vocab_size(),
        "fertility": round(fertility, 3),
        "tokens_per_compound": round(
            sum(len(v) for v in splits.values()) / len(splits), 2
        ),
        "german_words_per_256_tokens": round(words_per_256, 1),
        "compound_splits": splits,
    }


def report(m: dict) -> None:
    print(f"\n{'=' * 62}")
    print(f"vocab size                      {m['vocab_size']:,}")
    print(f"fertility (tokens per word)     {m['fertility']}")
    print(f"avg tokens per compound         {m['tokens_per_compound']}")
    print(f"German words per 256 tokens     {m['german_words_per_256_tokens']}")
    print(f"{'-' * 62}")
    for word, pieces in m["compound_splits"].items():
        print(f"  {len(pieces):>2}  {word}")
        print(f"      {' · '.join(pieces)}")
    print("=" * 62)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=16384)
    ap.add_argument("--corpus", type=Path, default=ROOT / "data" / "corpus.txt")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if not args.corpus.exists():
        print(f"no corpus at {args.corpus} — run data/prepare.py first")
        return 1

    out = args.out or HERE / f"silbe-{args.vocab_size}.json"
    tok = train(args.corpus, args.vocab_size, out)

    m = measure(tok, args.corpus)
    report(m)

    stats = out.with_suffix(".stats.json")
    stats.write_text(json.dumps(m, ensure_ascii=False, indent=2))
    print(f"\nsaved {out}\n      {stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
