"""
Stage 4 — training.

The loop is short. Take a batch of German, ask the model to predict each next
token, measure how wrong it was, nudge every weight to be slightly less wrong,
repeat. Everything else in this file exists to make that survivable on a
fanless laptop.

    python train.py --config configs/small.json

Resuming is automatic: point it at the same run directory and it continues
from the last checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from model.transformer import Config, Silbe  # noqa: E402

ROOT = Path(__file__).parent

# Gradients occasionally arrive enormous — a rare batch, an unlucky
# initialisation — and one such step can undo hours of training by throwing
# the weights somewhere they cannot recover from. Clipping bounds the damage.
GRAD_CLIP = 1.0

# AdamW's second beta. 0.95 rather than the 0.999 default: language models
# train with fewer, larger steps than the vision workloads the default was
# tuned on, and a shorter memory adapts faster.
BETAS = [0.9, 0.95]
WEIGHT_DECAY = 0.1


class Data:
    """Random slices from one long stream of tokens.

    Memory-mapped rather than loaded. The training file is hundreds of
    megabytes and this machine is sharing 16 GB with a browser; mapping lets
    the OS page in only what each batch touches.
    """

    def __init__(self, path: Path, block_size: int, batch_size: int):
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        self.batch_size = batch_size
        if len(self.tokens) < block_size + 1:
            raise ValueError(f"{path} has too few tokens ({len(self.tokens)})")

    def __len__(self) -> int:
        return len(self.tokens)

    def batch(self, rng: np.random.Generator) -> tuple[mx.array, mx.array]:
        # Start positions are uniform over the whole stream. No epochs, no
        # shuffling of records — with hundreds of millions of tokens, sampling
        # randomly is simpler and gives the same coverage.
        hi = len(self.tokens) - self.block_size - 1
        starts = rng.integers(0, hi, size=self.batch_size)

        x = np.stack([self.tokens[s : s + self.block_size] for s in starts])
        # Targets are the inputs shifted by one. Position i predicts i+1, and
        # every position in the sequence contributes to the loss — which is
        # why a transformer learns so much faster than reading one token at a
        # time would suggest.
        y = np.stack([self.tokens[s + 1 : s + 1 + self.block_size] for s in starts])

        return mx.array(x.astype(np.int32)), mx.array(y.astype(np.int32))


def schedule(cfg: dict):
    """Warm up, then decay.

    Warmup exists because a randomly initialised model has no idea what it is
    doing, and full-size steps in the first minute can push it somewhere it
    never recovers from. Cosine decay at the end lets it settle precisely
    rather than bouncing around the minimum.
    """
    lr = cfg["learning_rate"]
    warmup = cfg["warmup_steps"]
    return optim.join_schedules(
        [
            optim.linear_schedule(lr * 0.02, lr, warmup),
            optim.cosine_decay(lr, cfg["max_steps"] - warmup, lr * 0.1),
        ],
        [warmup],
    )


def evaluate(model: Silbe, data: Data, rng: np.random.Generator, batches: int) -> float:
    """Loss on text the model has never trained on.

    This is the only honest signal. Training loss always falls — it is what is
    being optimised. When this stops falling while training loss continues,
    the model has switched from learning German to memorising Wikipedia.
    """
    model.eval()
    total = 0.0
    for _ in range(batches):
        x, y = data.batch(rng)
        total += model.loss(x, y).item()
    model.train()
    return total / batches


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "small.json")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--steps", type=int, default=None, help="override max_steps")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    if args.steps:
        cfg["max_steps"] = args.steps

    vocab = cfg["vocab_size"]
    train_bin = args.data_dir / f"train-{vocab}.bin"
    val_bin = args.data_dir / f"val-{vocab}.bin"
    for p in (train_bin, val_bin):
        if not p.exists():
            print(f"missing {p} — run data/encode.py first")
            return 1

    run = args.out or ROOT / "checkpoints" / cfg["name"]
    run.mkdir(parents=True, exist_ok=True)

    model = Silbe(Config.from_dict(cfg))
    mx.eval(model.parameters())

    train_data = Data(train_bin, cfg["block_size"], cfg["batch_size"])
    val_data = Data(val_bin, cfg["block_size"], cfg["batch_size"])

    tokens_per_step = cfg["batch_size"] * cfg["block_size"]
    print(f"{cfg['name']}  {model.n_params / 1e6:.2f}M params")
    print(f"train {len(train_data):,} tokens · val {len(val_data):,} tokens")
    print(f"{tokens_per_step:,} tokens/step · {cfg['max_steps']:,} steps "
          f"= {tokens_per_step * cfg['max_steps'] / 1e6:.0f}M tokens "
          f"({tokens_per_step * cfg['max_steps'] / len(train_data):.2f} epochs)\n")

    opt = optim.AdamW(learning_rate=schedule(cfg), betas=BETAS, weight_decay=WEIGHT_DECAY)
    loss_and_grad = nn.value_and_grad(model, lambda m, x, y: m.loss(x, y))

    start_step = 0
    latest = run / "latest.safetensors"
    state_file = run / "state.json"
    if latest.exists() and state_file.exists():
        model.load_weights(str(latest))
        start_step = json.loads(state_file.read_text())["step"]
        print(f"resuming from step {start_step:,}\n")

    rng = np.random.default_rng(1337 + start_step)
    history_file = run / "history.jsonl"
    best_val = float("inf")

    # Ctrl-C should checkpoint rather than discard the run.
    stopping = {"now": False}
    signal.signal(signal.SIGINT, lambda *_: stopping.update(now=True))

    t0 = time.time()
    running = None

    for step in range(start_step + 1, cfg["max_steps"] + 1):
        x, y = train_data.batch(rng)

        loss, grads = loss_and_grad(model, x, y)
        grads, gnorm = optim.clip_grad_norm(grads, GRAD_CLIP)
        opt.update(model, grads)
        # MLX is lazy — nothing above has actually run until something forces
        # it. Evaluating here keeps the graph from growing without bound.
        mx.eval(model.parameters(), opt.state)

        l = loss.item()
        if not math.isfinite(l):
            print(f"\nstep {step}: loss is {l} — stopping. "
                  f"Lower the learning rate and resume from the last checkpoint.")
            return 1
        running = l if running is None else 0.9 * running + 0.1 * l

        if step % 10 == 0:
            el = time.time() - t0
            done = step - start_step
            rate = done / el
            eta = (cfg["max_steps"] - step) / rate if rate else 0
            print(f"\rstep {step:>6}/{cfg['max_steps']}  loss {running:.3f}  "
                  f"lr {opt.learning_rate.item():.2e}  |g| {gnorm.item():.2f}  "
                  f"{rate * tokens_per_step / 1000:.0f}k tok/s  "
                  f"eta {eta / 60:.0f}m   ", end="", flush=True)

        if step % cfg["eval_every"] == 0 or step == cfg["max_steps"]:
            val = evaluate(model, val_data, np.random.default_rng(7), cfg["eval_batches"])
            print(f"\rstep {step:>6}  train {running:.3f}  val {val:.3f}"
                  f"  ppl {math.exp(min(val, 20)):.1f}" + " " * 30)

            with history_file.open("a") as f:
                f.write(json.dumps({"step": step, "train": running, "val": val,
                                    "elapsed": round(time.time() - t0, 1)}) + "\n")

            # Kept separately from the latest checkpoint: the last step is not
            # always the best one, and overfitting is only visible in hindsight.
            if val < best_val:
                best_val = val
                model.save_weights(str(run / "best.safetensors"))

        if step % cfg["checkpoint_every"] == 0 or stopping["now"] or step == cfg["max_steps"]:
            model.save_weights(str(latest))
            state_file.write_text(json.dumps({"step": step, "best_val": best_val}))
            if stopping["now"]:
                print(f"\n\ninterrupted — saved at step {step}. "
                      f"Rerun the same command to continue.")
                return 0

    mins = (time.time() - t0) / 60
    print(f"\ndone in {mins:.0f}m · best val {best_val:.3f}")
    print(f"weights in {run}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
