# Silbe — design

A German language model trained from scratch on a MacBook Air.

Status: design. No code written yet.

---

## What this is

`Silbe` learns German by reading German. It starts knowing nothing — not that
words exist, not that spaces separate them, not that `der` and `die` are
different. Random numbers. Everything it ends up knowing, it works out from
text.

The name is German for *syllable*, which is what the tokenizer does: cut
language into pieces small enough for a machine to count.

## What it is not

It is **not a chatbot** and will never be one. At 15 million parameters there
is no capacity for facts, reasoning or instruction-following. Asked a question
it will produce plausible German that answers nothing.

That is the correct outcome, not a failure. A model this size can be good at
*continuing German text*. Anything more is a different project with a
different budget.

Naming this now, because the temptation to judge it against ChatGPT will be
strong the first time it produces a sentence.

## Why build it

Two reasons, in order of honesty.

**To understand what happens underneath.** Wayfare consumes five model
providers competently and explains nothing about how any of them work. This is
the other half.

**Because German is the interesting case.** English is the default in every
tutorial, and English is easy: mostly short words, few endings, forgiving word
order. German has four cases, adjective endings that agree with both, verbs
that move to the end of subordinate clauses, and compound nouns of arbitrary
length.

`Arbeitsunfähigkeitsbescheinigung` is one word. How a tokenizer handles that —
and what it costs when it handles it badly — is a real engineering question
with measurable answers. See *The headline experiment* below.

## The hardware, which decides everything

MacBook Air, Apple M5, 16 GB unified memory, 8 GPU cores.

Three consequences that shape every decision in this document:

**Unified memory is the advantage.** The GPU addresses the same 16 GB as the
CPU, so there is no separate VRAM ceiling to fight. A discrete laptop GPU with
8 GB would be worse for this.

**No fan is the constraint.** Sustained load makes the M5 throttle to stay
within thermal limits. A run that benchmarks well for ten minutes can be
30–40% slower across six hours. Every long run must checkpoint frequently,
because thermal shutdown, a closed lid or a crash must cost minutes rather
than a night.

**16 GB is the ceiling.** Model weights, gradients and optimiser state all sit
in memory at once — roughly four times the parameter count in bytes, before
activations. This is why 40M is the practical maximum and 100M is not
attempted.

Framework: **MLX**, Apple's array framework. It targets Metal directly and is
the fastest option on this chip. PyTorch's MPS backend works but is slower and
has more gaps.

---

## The pipeline

Five stages. Each writes a file the next reads, so a mistake late does not
cost the work done early.

```
1. GET TEXT      German Wikipedia dump → cleaned plain text
2. TOKENIZE      learn to cut German into ~16k pieces
3. ENCODE        all text → one flat array of token IDs
4. TRAIN         the long part. loss falls. checkpoints saved.
5. GENERATE      give it a prompt, read what comes back
```

Stages 1–3 run once and take under an hour. Stage 4 is measured in hours.
Stage 5 runs constantly, because reading the output is how you find out
whether anything worked.

### 1. Get text

Source: the German Wikipedia dump (`dewiki`), CC BY-SA licensed, so the
resulting model can be published without a licensing problem.

The dump is XML with wiki markup. It needs stripping to plain prose:
templates, tables, infoboxes, reference markers, category links — all removed.
What survives is article text.

Target for the first run: **roughly 500 MB of clean German**, which is a slice
of the full dump rather than all of it. Enough to train 15M without
repetition; small enough to prepare in minutes.

Wikipedia is used here as a *language sample*, not as a source of facts. The
model learns grammar, morphology and vocabulary. Whether any given article is
factually correct is irrelevant to what is being extracted — a false sentence
teaches the same grammar as a true one. This is worth stating because the
standard advice about Wikipedia concerns citation, which is a different use.

### 2. Tokenize

Text has to become numbers. A tokenizer decides how.

We train a **byte-pair encoding (BPE)** tokenizer on the corpus. BPE starts
from single characters and repeatedly merges the most frequent adjacent pair,
building up subword units. Common words end up as single tokens; rare ones
break into pieces.

Vocabulary size for the first run: **16,384**.

This choice matters more in German than in English, and it is the subject of
the headline experiment below.

Output: a vocabulary file plus merge rules. Deterministic — the same text
always encodes the same way.

### 3. Encode

Run the tokenizer over the whole corpus once and write the token IDs to a flat
binary file as `uint16` (valid because vocab < 65,536).

Why a separate stage: tokenizing is slow and the trainer will read this data
thousands of times. Doing it once turns a repeated cost into a fixed one, and
the trainer then does nothing but memory-map an array and slice it.

Split: **99% train, 1% validation.** The validation slice is never trained on
and exists solely to detect overfitting.

### 4. Train

A decoder-only transformer — the same family as GPT, Llama and Mistral,
scaled down by three orders of magnitude.

One step:

1. Take a batch of sequences from the encoded data
2. For every position, predict the next token
3. Compare predictions to truth → **cross-entropy loss**
4. Compute gradients for all parameters
5. Nudge every parameter to reduce loss

Repeated for tens of thousands of steps.

**Optimiser:** AdamW. **Learning rate:** linear warmup over the first ~200
steps, then cosine decay. Warmup exists because a randomly initialised model
takes destructively large steps if allowed to; several hundred steps of
restraint prevent it wrecking itself in the first minute.

**Expected loss trajectory** for a 16k vocabulary:

| Loss | Meaning |
|------|---------|
| ~9.7 | Uniform random — the starting point |
| ~6   | Learned which tokens are common |
| ~4.5 | Real German words, no coherence |
| ~3.5 | Grammatical fragments |
| ~3.0 | Coherent sentences |

**Overfitting** is detected by watching train and validation loss together.
While both fall, the model is learning German. When training loss keeps
falling and validation loss turns upward, it has begun memorising Wikipedia
rather than learning the language. That inflection is the signal to stop.

**Checkpoints** every 500 steps, keeping the best validation loss separately
from the latest. Non-negotiable on a fanless machine.

### 5. Generate

Load a checkpoint, give it a prompt, sample tokens one at a time.

Sampling controls, adjustable without retraining:

- **temperature** — 0.1 is repetitive and safe, 1.5 is unhinged
- **top-k / top-p** — restrict choices to the most likely tokens
- **repetition penalty** — discourage loops, which small models fall into
  readily

---

## Model architecture

Standard decoder-only transformer. Nothing exotic — the point is to understand
the standard thing, not to invent a variant.

Per block: masked multi-head self-attention, then a feed-forward network, each
with a residual connection and pre-normalisation (RMSNorm). Rotary position
embeddings (RoPE) for position, weight tying between input embedding and
output projection.

## The size ladder

Size lives entirely in configuration. Five numbers separate the smallest model
from the largest; no structural code changes.

| Config  | Layers | Heads | Width | Context | Params | Est. time  | Purpose |
|---------|--------|-------|-------|---------|--------|------------|---------|
| `nano`  | 4      | 4     | 128   | 128     | ~2M    | 5 min      | Smoke test |
| `small` | 8      | 8     | 384   | 256     | ~15M   | 2–4 h      | First real model |
| `base`  | 12     | 12    | 512   | 512     | ~40M   | overnight  | Best quality |

`nano` is not a model anyone keeps. It exists so the entire pipeline can be
proven end to end in five minutes, because discovering a data-loader bug at
hour three of a four-hour run is an avoidable way to lose an afternoon.

Times are estimates to be replaced with measurements after the first run.

The same configuration file would run on a rented GPU unchanged. Nothing here
assumes Apple Silicon except the framework.

---

## The headline experiment

Training three sizes and plotting loss against parameters reproduces the
scaling relationship the field is built on. Worth doing, and the graph is
good, but it is not the interesting finding.

**The interesting finding is about German and tokenizers.**

German builds arbitrarily long compound nouns. `Hauptbahnhof`,
`Geschwindigkeitsbegrenzung`, `Arbeitsunfähigkeitsbescheinigung`. A BPE
tokenizer with a small vocabulary has no choice but to shatter these into many
short fragments; a larger vocabulary keeps more of them intact.

That trade-off is measurable, and both directions cost something:

- **Small vocab** — fewer embedding parameters, but every sentence becomes
  more tokens, so the same context window holds less actual German and each
  training step covers less text
- **Large vocab** — fewer tokens per sentence, but the embedding table grows
  and rare tokens are seen too rarely to learn well

The experiment: train the same `small` config with vocabularies of **4k, 8k,
16k and 32k**, and measure

1. **Fertility** — average tokens per German word
2. **Compound integrity** — how a fixed list of long compounds gets split
3. **Effective context** — how many German words fit in 256 tokens
4. **Final validation loss**, and loss per unit of wall-clock time

This produces a defensible answer to a real question — *what vocabulary size
should a German model use, and what does getting it wrong cost?* — measured
rather than assumed.

It is the same shape as the JobPulse truncation finding: one parameter, chosen
carelessly, quietly degrading everything downstream.

---

## Repository layout

```
silbe/
  configs/         nano.json, small.json, base.json
  data/            prepare.py     dump → clean text
                   encode.py      text → token IDs
  tokenizer/       train.py       learn BPE
                   silbe.py       encode/decode
  model/           transformer.py the architecture
  train.py         the training loop
  generate.py      sampling
  eval/            tokenizer_study.py   the headline experiment
                   scaling.py           loss vs parameters
  checkpoints/     gitignored — weights are large
  docs/            this file, plus findings as they arrive
```

Large artefacts — dumps, encoded arrays, checkpoints — stay out of git.

## What done looks like

1. A trained 15M model that produces recognisable German
2. A 40M model that produces coherent German sentences
3. The tokenizer study, with numbers and a plot
4. The scaling curve across three sizes
5. A README that explains what was learned, not what was built
6. Optionally: served locally and wired into Wayfare as a fallback provider

## Risks, honestly

**Thermal throttling makes long runs slower than estimated.** Mitigated by
checkpointing and by measuring rather than trusting the estimates above.

**15M may be too small to produce anything satisfying.** Possible. `nano`
existing means this is discovered in minutes rather than hours, and the ladder
means the answer is to go up a rung rather than to start over.

**Wikipedia markup cleaning is fiddly** and produces subtly bad text if rushed
— stray templates, reference numbers mid-sentence. The model will happily
learn to reproduce them. Inspecting the cleaned text by eye before encoding is
part of the work, not an optional check.

**Scope creep towards a chatbot.** It will not become one. The instruction
tuning, RLHF and data volume required are all out of reach here, and pretending
otherwise would waste the project.
