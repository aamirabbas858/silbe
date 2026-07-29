"""
The scaling curve: what does model size buy, holding everything else fixed?

Three models trained identically — same 531 MB of German Wikipedia, same
256-token context, same batch, same 12,000 steps, same 98M tokens. The only
variable is width and depth.

Loss against parameters follows a power law, L ~ N^-a. Fitting it on three
points from a laptop is not a contribution to the literature; reproducing the
shape at all, from scratch, is the point.
"""
import json, math, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import mlx.core as mx
from model.transformer import Silbe, Config

ROOT = Path(__file__).parent.parent
MODELS = ["tiny", "mid", "small"]

pts = []
for n in MODELS:
    cfg = json.loads((ROOT / "configs" / f"{n}.json").read_text())
    m = Silbe(Config.from_dict(cfg)); mx.eval(m.parameters())
    hist = [json.loads(l) for l in (ROOT / "checkpoints" / n / "history.jsonl").open()]
    pts.append((n, m.n_params, min(d["val"] for d in hist), hist))

N = np.array([p[1] for p in pts], dtype=float)
L = np.array([p[2] for p in pts], dtype=float)

# Fit log L = log c - a log N
a, logc = np.polyfit(np.log(N), np.log(L), 1)
a = -a

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

# left: the curve
ax1.loglog(N, L, "o", ms=9, color="#c0392b", zorder=3)
xs = np.logspace(np.log10(N[0] * 0.7), np.log10(N[-1] * 1.5), 50)
ax1.loglog(xs, np.exp(logc) * xs ** (-a), "--", lw=1.2, color="#7f8c8d",
           label=f"$L \\propto N^{{-{a:.3f}}}$")
for n, p, l, _ in pts:
    ax1.annotate(f"{n}\n{p/1e6:.1f}M", (p, l), textcoords="offset points",
                 xytext=(8, 8), fontsize=9)
ax1.set_xlabel("parameters"); ax1.set_ylabel("validation loss")
ax1.set_title("Loss falls as a power law in size", fontsize=11)
ax1.legend(frameon=False); ax1.grid(alpha=.25, which="both")

# right: the training curves that produced those points
for (n, p, l, hist), c in zip(pts, ["#95a5a6", "#e67e22", "#c0392b"]):
    steps = [d["step"] for d in hist]; vals = [d["val"] for d in hist]
    ax2.plot(steps, vals, color=c, lw=1.6, label=f"{n} ({p/1e6:.1f}M)")
ax2.set_xlabel("step"); ax2.set_ylabel("validation loss")
ax2.set_title("Same data, same steps, different size", fontsize=11)
ax2.legend(frameon=False); ax2.grid(alpha=.25)

fig.tight_layout()
out = ROOT / "docs" / "figures" / "scaling.png"
fig.savefig(out, dpi=150)

print(f"{'model':6} {'params':>10} {'val':>8} {'ppl':>7}")
for n, p, l, _ in pts:
    print(f"{n:6} {p/1e6:9.2f}M {l:8.4f} {math.exp(l):7.1f}")
print(f"\nfitted exponent a = {a:.4f}   (Kaplan et al. 2020 report ~0.076 for parameters)")
print(f"saved {out}")
