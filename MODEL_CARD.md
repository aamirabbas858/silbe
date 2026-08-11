---
language:
  - de
license: apache-2.0
library_name: mlx
tags:
  - german
  - from-scratch
  - transformer
  - tokenization
  - mlx
  - apple-silicon
datasets:
  - wikimedia/wikipedia
pipeline_tag: text-generation
inference: false
---

# Silbe (small) — a 20M-parameter German language model trained from scratch

*Silbe* is German for **syllable**, which is roughly what a tokenizer produces:
language cut into pieces small enough for a machine to count.

No pretrained weights, no fine-tuning, no distillation. The model was trained
from random initialisation on 531 MB of German Wikipedia. Everything it knows
about German it derived from that text.

**This model exists to demonstrate a result about tokenizer evaluation, not to
be used in an application.** Read the limitations before anything else.

## What it is

| | |
|---|---|
| Parameters | 20.45M |
| Layers / heads / width | 8 / 8 / 384 |
| Context | 256 tokens |
| Vocabulary | 16,384 (byte-level BPE, trained on the same corpus) |
| Attention | causal multi-head, RoPE on queries and keys |
| Normalisation | RMSNorm, pre-norm |
| Feed-forward | SwiGLU, hidden width 8/3 × d_model |
| Output | weight tying — embedding reused as output projection |
| Training | 12,000 steps, AdamW, linear warmup → cosine decay to 10% |
| Validation loss | 3.7623 (perplexity 43.0) |
| Framework | MLX (Apple Silicon / Metal) |

Deliberately the standard decoder-only arrangement — the GPT/Llama/Mistral
family scaled down by three orders of magnitude. The goal was to understand
what everyone else builds, not to invent a variant.

## Limitations — read this part

**It is not a chatbot and cannot become one.** At 20M parameters there is no
capacity for facts, reasoning or instruction-following. Asked a question it
produces fluent German that answers nothing.

**The content it generates is nonsense.** The grammar is not. What holds up is
case agreement through nested phrases, relative clauses with the verb correctly
final, passive voice, and compound nouns built correctly from real morphemes
even when the compound itself is invented. What does not hold up is any claim
about the world.

**Wikipedia was used as a language sample, not a source of facts.** The model
learned morphology, syntax and register. Whether any source article was
factually correct is irrelevant to what was extracted, and the model should
never be treated as a source of information about anything.

**Apple Silicon required to run as published.** MLX targets Metal. The weights
are stored as `safetensors`, so the tensors themselves are portable, but tensor
names follow this repository's module layout and a PyTorch user would need a
matching module definition. The architecture is standard, so that port is
mechanical rather than a redesign.

**Context is 256 tokens.** Roughly 220 German words.

**No safety tuning of any kind.** The model reproduces the distribution of
German Wikipedia, including whatever biases that corpus carries.

## The result this model exists to demonstrate

Four tokenizers were trained on the same corpus at 4,096 / 8,192 / 16,384 /
32,768 vocabulary, each with an identical model behind it.

| vocab | params | embedding share | tokens/word | val loss | bits/char |
|---|---|---|---|---|---|
| 4,096 | 5.07M | 21% | 2.373 | **3.678** ← "best" | 1.6109 |
| 8,192 | 6.11M | 34% | 2.087 | 4.171 | 1.6079 |
| 16,384 | 8.21M | 51% | 1.856 | 4.551 | **1.5674** ← best |
| 32,768 | 12.41M | 68% | 1.680 | **5.058** ← "worst" | 1.5850 |

**The two orderings are exact opposites.** Validation loss is computed per
token against the vocabulary, so a model choosing among 4,096 options faces an
easier problem than one choosing among 32,768 — it records a lower number
without writing better German. Bits per character removes that distortion by
scoring every model on the same text in the same unit:

```
bpc = (loss_in_nats / ln 2) / characters_per_token
```

Measured that way the best vocabulary is 16,384, and the reason it stops
improving past that is the `embedding share` column. At 4,096 the token table
is 21% of the model; at 32,768 it is 68% — two thirds of the parameters spent
storing what tokens are rather than processing language.

That validation loss is not comparable across vocabulary sizes is known to
people who work on tokenizers. What this repository adds is a controlled
demonstration with the parameter-budget explanation attached.

## Sibling checkpoints

Same corpus, same context, same 12,000 steps, differing only in width and depth:

| model | params | layers | heads | width | val loss | perplexity |
|---|---|---|---|---|---|---|
| tiny | 2.95M | 4 | 4 | 128 | 4.5045 | 90.4 |
| mid | 8.21M | 5 | 4 | 256 | 4.0396 | 56.8 |
| **small** | **20.45M** | **8** | **8** | **384** | **3.7623** | **43.0** |

Fitting `log L` against `log N` gives an exponent of 0.0933, against ≈0.076
reported by [Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361) for
parameters. **That agreement is closer than three points deserve** — the fit is
weak and these models are undertrained relative to their size. The claim is
only that the shape appears at all at this scale.

## Usage

```python
# Apple Silicon, MLX
import mlx.core as mx
from model.transformer import Transformer   # from the source repository
from tokenizers import Tokenizer

tok = Tokenizer.from_file("tokenizer/silbe-16384.json")
model = Transformer(vocab_size=16384, n_layers=8, n_heads=8,
                    n_embd=384, block_size=256)
model.load_weights("best.safetensors")
```

Generation supports temperature, top-k, top-p and a repetition penalty. See
`generate.py` in the source repository.

## Training data

German Wikipedia, CC BY-SA. 531 MB of cleaned prose after two filters: articles
under 600 characters are stubs and disambiguation pages; articles where over
half the lines are short are lists rather than prose. 29,870 articles kept,
10,130 dropped. Encoded to 126.9M token IDs.

The train/validation split is **by position, not random** — articles are
contiguous, so a shuffled split would put one half of an article in training and
the other in validation, and the model would score well for the wrong reason.

## Licence

Apache-2.0 for the code and weights. The training corpus is German Wikipedia
under CC BY-SA and is attributed accordingly. Whether model weights constitute a
derivative work of their training data is unsettled; this release follows common
practice rather than a settled rule.

## Citation

```bibtex
@software{aamir_silbe_2026,
  author = {Aamir, Abbas},
  title  = {Silbe: a German language model trained from scratch,
            with a study of tokenizer vocabulary size},
  year   = {2026},
  url    = {https://github.com/aamirabbas858/silbe}
}
```
