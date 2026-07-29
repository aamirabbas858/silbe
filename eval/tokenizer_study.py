"""
What vocabulary size should a German model use, and what does getting it
wrong cost?

German welds words together — Geschwindigkeitsbegrenzung, not "speed limit".
A BPE tokenizer has to cut somewhere, and vocabulary size decides where. Both
directions cost something:

  small vocab   fewer embedding parameters, but every sentence becomes more
                tokens, so a fixed context window holds less German and each
                training step covers less language

  large vocab   fewer tokens per sentence, but the embedding table grows and
                rare tokens are seen too rarely to learn well

This trains one tokenizer and one model per vocabulary size and measures both
ends of that trade.

    python eval/tokenizer_study.py

Resumable: anything already on disk is reused, so an interrupted run costs
only the step it was on.

--- the comparison that would have been wrong ---

Validation loss is NOT comparable across vocabulary sizes. Cross-entropy is
measured per token, over the vocabulary; predicting one of 4,096 options is
inherently easier than one of 32,768, so a small vocabulary posts a lower
loss while being no better at German.

Bits per character removes both distortions at once — it converts nats to
bits and divides by characters rather than tokens, so every model is scored
on the same text in the same unit regardless of how it chopped it up.

    bpc = (loss_in_nats / ln 2) / characters_per_token

This is the number the conclusion rests on. Loss is reported alongside it
only to show how misleading it would have been.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

VOCABS = [4096, 8192, 16384, 32768]

# The study uses the mid architecture rather than small: it is 2.5x faster and
# the question is about differences between vocabularies, not absolute quality.
ARCH = "mid"
STEPS = 4000

PY = str(ROOT / ".venv" / "bin" / "python")
CORPUS = ROOT / "data" / "corpus.txt"


def run(cmd: list[str]) -> None:
    print(f"  $ {' '.join(str(c) for c in cmd[1:])}", flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> int:
    results = []
    # Character count of the corpus, needed to convert loss into bits per
    # character. Read once — it does not change between vocabularies.
    chars = CORPUS.stat().st_size

    for vocab in VOCABS:
        print(f"\n{'=' * 60}\nvocab {vocab:,}\n{'=' * 60}", flush=True)

        tok_path = ROOT / "tokenizer" / f"silbe-{vocab}.json"
        if not tok_path.exists():
            run([PY, "tokenizer/train.py", "--vocab-size", str(vocab)])

        train_bin = ROOT / "data" / f"train-{vocab}.bin"
        if not train_bin.exists():
            run([PY, "data/encode.py", "--tokenizer", str(tok_path)])

        # One config per vocabulary — identical but for vocab_size, so the
        # embedding table is the only structural difference.
        cfg = json.loads((ROOT / "configs" / f"{ARCH}.json").read_text())
        cfg.update(name=f"vocab{vocab}", vocab_size=vocab, max_steps=STEPS,
                   eval_every=500, checkpoint_every=1000)
        cfg_path = ROOT / "configs" / f"_vocab{vocab}.json"
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")

        hist_path = ROOT / "checkpoints" / f"vocab{vocab}" / "history.jsonl"
        if not hist_path.exists():
            run([PY, "-u", "train.py", "--config", str(cfg_path)])

        # --- measurements -------------------------------------------------
        tok = Tokenizer.from_file(str(tok_path))
        stats = json.loads((tok_path.with_suffix(".stats.json")).read_text())

        n_tokens = train_bin.stat().st_size // 2 + (
            (ROOT / "data" / f"val-{vocab}.bin").stat().st_size // 2
        )
        chars_per_token = chars / n_tokens

        hist = [json.loads(l) for l in hist_path.open()]
        val = min(d["val"] for d in hist)
        bpc = (val / math.log(2)) / chars_per_token

        import mlx.core as mx
        from model.transformer import Config, Silbe

        m = Silbe(Config.from_dict(cfg))
        mx.eval(m.parameters())
        embed_params = vocab * cfg["n_embd"]

        results.append({
            "vocab": vocab,
            "params": m.n_params,
            "embed_params": embed_params,
            "embed_share": embed_params / m.n_params,
            "fertility": stats["fertility"],
            "tokens_per_compound": stats["tokens_per_compound"],
            "words_per_256": stats["german_words_per_256_tokens"],
            "chars_per_token": round(chars_per_token, 3),
            "val_loss": round(val, 4),
            "bits_per_char": round(bpc, 4),
        })
        print(json.dumps(results[-1], indent=2), flush=True)

    out = ROOT / "docs" / "tokenizer_study.json"
    out.write_text(json.dumps(results, indent=2))
    report(results)
    plot(results)
    return 0


def report(r: list[dict]) -> None:
    print(f"\n{'=' * 78}")
    print(f"{'vocab':>7} {'params':>9} {'embed%':>7} {'fert':>6} {'w/256':>7} "
          f"{'val loss':>9} {'bits/char':>10}")
    print("-" * 78)
    for d in r:
        print(f"{d['vocab']:>7,} {d['params']/1e6:>8.2f}M {d['embed_share']*100:>6.0f}% "
              f"{d['fertility']:>6.3f} {d['words_per_256']:>7.0f} "
              f"{d['val_loss']:>9.4f} {d['bits_per_char']:>10.4f}")
    print("=" * 78)

    best_loss = min(r, key=lambda d: d["val_loss"])
    best_bpc = min(r, key=lambda d: d["bits_per_char"])
    print(f"\nlowest val loss : vocab {best_loss['vocab']:,}")
    print(f"lowest bits/char: vocab {best_bpc['vocab']:,}   <- the honest winner")
    if best_loss["vocab"] != best_bpc["vocab"]:
        print("\nThese disagree, which is the point: loss rewards a small vocabulary "
              "for having fewer options to choose between, not for writing better "
              "German.")


def plot(r: list[dict]) -> None:
    v = [d["vocab"] for d in r]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    axes[0].semilogx(v, [d["fertility"] for d in r], "o-", color="#c0392b", base=2)
    axes[0].set_ylabel("tokens per German word")
    axes[0].set_title("Bigger vocabulary, fewer tokens per word", fontsize=10)

    axes[1].semilogx(v, [d["bits_per_char"] for d in r], "o-", color="#2c3e50", base=2,
                     label="bits/char (comparable)")
    ax1b = axes[1].twinx()
    ax1b.semilogx(v, [d["val_loss"] for d in r], "s--", color="#95a5a6", base=2,
                  label="val loss (not comparable)")
    ax1b.set_ylabel("val loss", color="#95a5a6")
    axes[1].set_ylabel("bits per character")
    axes[1].set_title("Loss disagrees with the honest metric", fontsize=10)

    axes[2].semilogx(v, [d["embed_share"] * 100 for d in r], "o-", color="#e67e22", base=2)
    axes[2].set_ylabel("% of parameters in the embedding")
    axes[2].set_title("What a big vocabulary costs", fontsize=10)

    for ax in axes:
        ax.set_xlabel("vocabulary size")
        ax.set_xticks(v)
        ax.set_xticklabels([f"{x//1024}k" for x in v])
        ax.grid(alpha=.25)

    fig.tight_layout()
    out = ROOT / "docs" / "figures" / "tokenizer_study.png"
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    sys.exit(main())
