---
language:
  - de
license: apache-2.0
library_name: tokenizers
tags:
  - german
  - tokenizer
  - bpe
  - byte-level-bpe
  - morphology
datasets:
  - wikimedia/wikipedia
---

# Silbe German tokenizers — 4k / 8k / 16k / 32k

Four byte-level BPE tokenizers trained on the same 531 MB of German Wikipedia
prose, differing only in vocabulary size. Each ships with measured statistics so
you can pick one on evidence rather than habit.

*Silbe* is German for **syllable** — what a tokenizer produces: language cut
into pieces small enough for a machine to count.

Standard `tokenizers` JSON. No framework dependency — usable from PyTorch, JAX,
MLX or plain `tokenizers`.

```python
from tokenizers import Tokenizer
tok = Tokenizer.from_file("silbe-16384.json")
tok.encode("Arbeitsunfähigkeitsbescheinigung").tokens
# ['Arbeits', 'un', 'fähigkeit', 's', 'be', 'schein', 'igung']
```

## Which one to use

| vocab | tokens per word | tokens per compound | German words per 256 tokens |
|---|---|---|---|
| 4,096 | 2.373 | 7.89 | 167.7 |
| 8,192 | 2.087 | 6.56 | 194.6 |
| **16,384** | **1.856** | **5.44** | **211.5** |
| 32,768 | 1.680 | 4.33 | 221.1 |

Fertility falls monotonically — a larger vocabulary always cuts German into
fewer pieces, and a 256-token window holds 32% more language at 32k than at 4k.

**That does not mean 32,768 is the right choice.** In the accompanying study,
where an identical model was trained behind each of these tokenizers, quality
measured in bits per character peaked at **16,384** and got worse at 32,768. The
reason is the parameter budget: at 4k the embedding table is 21% of the model,
at 32k it is 68%, so two thirds of the parameters go on storing what tokens are
rather than processing language.

If you are training a small model, 16,384 is the one to reach for. At larger
model sizes the embedding share shrinks and the balance shifts — this optimum is
specific to the scale it was measured at, and should not be extrapolated.

## A warning about comparing tokenizers by loss

Validation loss is computed per token, against the vocabulary. A model choosing
among 4,096 options faces an easier problem than one choosing among 32,768, so a
small vocabulary posts a lower loss while producing worse German. In the study,
validation loss and bits per character ranked these four in **exactly opposite
orders**.

If you are selecting a tokenizer, normalise to a per-character unit:

```
bpc = (loss_in_nats / ln 2) / characters_per_token
```

This is known to people who work on tokenizers, but it is easy to hit by
accident. Full tables and figures: https://github.com/aamirabbas858/silbe

## Why German makes this interesting

German welds words together, so where BPE cuts is a real engineering decision
with a measurable cost. At 16,384:

```
 1  Hauptbahnhof
    Hauptbahnhof

 2  Krankenversicherung
    Kranken · versicherung

 5  Donaudampfschifffahrtsgesellschaft
    Donau · dampf · schiff · fahrts · gesellschaft

 7  Arbeitsunfähigkeitsbescheinigung
    Arbeits · un · fähigkeit · s · be · schein · igung

14  Rindfleischetikettierungsüberwachungsaufgabenübertragungsgesetz
    R · ind · fl · eisch · etik · ett · ierungs · über · wach ·
    ungs · aufgaben · über · tragungs · gesetz
```

`Hauptbahnhof` survives as a single token — frequent enough in 531 MB of German
to earn its own vocabulary slot. Nobody specified that; it fell out of the
merges. `Donaudampfschifffahrtsgesellschaft` splits into its five real
constituents. The legal compound at the bottom does not, because it is rare
enough that BPE never learned its parts.

Per-tokenizer splits for all nine test compounds are in the `.stats.json` files.

## Files

| file | contents |
|---|---|
| `silbe-{4096,8192,16384,32768}.json` | the tokenizers |
| `silbe-*.stats.json` | fertility, compound splits, words per window |
| `train.py` | the script that produced them |

## Training data

German Wikipedia, CC BY-SA. 531 MB of cleaned prose: articles under 600
characters dropped as stubs and disambiguation pages, articles where over half
the lines are short dropped as lists rather than prose. 29,870 kept, 10,130
dropped.

## Licence

Apache-2.0. Corpus is German Wikipedia under CC BY-SA, attributed accordingly.

## Citation

```bibtex
@software{aamir_silbe_2026,
  author = {Aamir, Abbas},
  title  = {Silbe: German byte-level BPE tokenizers and a study of
            vocabulary size},
  year   = {2026},
  url    = {https://github.com/aamirabbas858/silbe}
}
```
