"""
Stage 5 — sampling.

Give the model some German, let it continue. This is where you find out
whether any of the training worked, and it is the only stage worth running
repeatedly.

    python generate.py --config configs/small.json --prompt "Berlin ist"

The sampling settings change the output completely without retraining
anything, so they are worth playing with before concluding the model is bad.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent))
from model.transformer import Config, Silbe  # noqa: E402

ROOT = Path(__file__).parent


def sample_next(
    logits: mx.array, temperature: float, top_k: int, top_p: float, seen: set[int],
    repetition_penalty: float,
) -> int:
    """Pick one token from the model's scores over the whole vocabulary."""
    logits = logits.astype(mx.float32)

    # Small models fall into loops — the same clause repeated forever. Pushing
    # down tokens already used is a blunt fix, but it is the difference
    # between readable output and a stuck record.
    if repetition_penalty != 1.0 and seen:
        idx = mx.array(sorted(seen))
        vals = logits[idx]
        # Divide positive scores, multiply negative ones: both move the token
        # towards less likely regardless of sign.
        logits[idx] = mx.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)

    if temperature <= 0:
        return int(mx.argmax(logits).item())  # greedy: always the top choice

    logits = logits / temperature

    # top-k: consider only the k most likely tokens. Everything else is
    # discarded before the dice are rolled, which stops the long tail of
    # nonsense from ever being chosen.
    if top_k > 0:
        kth = mx.sort(logits)[-top_k]
        logits = mx.where(logits < kth, -mx.inf, logits)

    probs = mx.softmax(logits, axis=-1)

    # top-p (nucleus): keep the smallest set of tokens whose probability sums
    # past p. Unlike top-k this adapts — when the model is confident it
    # considers few options, when unsure it considers many.
    if 0 < top_p < 1:
        order = mx.argsort(-probs)
        ordered = probs[order]
        cumulative = mx.cumsum(ordered)
        # Always keep the first token, or a very confident step could keep none.
        keep = cumulative - ordered < top_p
        ordered = mx.where(keep, ordered, 0.0)
        probs = mx.zeros_like(probs)
        probs[order] = ordered / ordered.sum()

    return int(mx.random.categorical(mx.log(probs + 1e-10)).item())


def generate(
    model: Silbe, tok: Tokenizer, prompt: str, n: int, temperature: float,
    top_k: int, top_p: float, repetition_penalty: float, stream: bool,
) -> str:
    ids = tok.encode(prompt).ids if prompt else [0]
    block = model.cfg.block_size
    seen: set[int] = set(ids)

    if stream:
        print(prompt, end="", flush=True)

    out = list(ids)
    for _ in range(n):
        # The model can only see block_size tokens, so long generations feed
        # it a sliding window. Everything older is simply forgotten — there is
        # no memory beyond the context.
        window = mx.array([out[-block:]], dtype=mx.int32)
        logits = model(window)[0, -1]
        nxt = sample_next(logits, temperature, top_k, top_p, seen, repetition_penalty)

        out.append(nxt)
        seen.add(nxt)

        if stream:
            # Decode incrementally so multi-token characters (umlauts are two
            # bytes) print correctly rather than as fragments.
            print(tok.decode(out[len(ids):])[len(tok.decode(out[len(ids):-1])):],
                  end="", flush=True)

    return tok.decode(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "small.json")
    ap.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizer" / "silbe-16384.json")
    ap.add_argument("--weights", type=Path, default=None, help="defaults to best checkpoint")
    ap.add_argument("--prompt", default="Berlin ist")
    ap.add_argument("--n", type=int, default=200, help="tokens to generate")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repetition-penalty", type=float, default=1.15)
    ap.add_argument("--samples", type=int, default=1)
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    weights = args.weights or ROOT / "checkpoints" / cfg["name"] / "best.safetensors"
    if not weights.exists():
        print(f"no weights at {weights} — train first")
        return 1

    model = Silbe(Config.from_dict(cfg))
    model.load_weights(str(weights))
    model.eval()
    mx.eval(model.parameters())

    tok = Tokenizer.from_file(str(args.tokenizer))

    for i in range(args.samples):
        if args.samples > 1:
            print(f"\n{'─' * 60}\nsample {i + 1}\n{'─' * 60}")
        t0 = time.time()
        generate(model, tok, args.prompt, args.n, args.temperature,
                 args.top_k, args.top_p, args.repetition_penalty, stream=True)
        print(f"\n\n[{args.n / (time.time() - t0):.0f} tok/s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
