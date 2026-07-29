"""
The model — a decoder-only transformer.

Same family as GPT, Llama and Mistral, scaled down by three orders of
magnitude. Nothing here is novel and that is deliberate: the goal is to
understand the standard architecture, not to invent a variant.

It does one thing. Given a sequence of tokens, for every position at once, it
produces a score for every token in the vocabulary — how likely each is to
come next. Training adjusts the weights so those scores match reality.

Four choices worth naming, all of them what current models do:

  RMSNorm rather than LayerNorm — cheaper, no mean subtraction, no bias, and
  no measured loss in quality.

  Pre-normalisation — normalise before each sub-layer rather than after. Deep
  post-norm stacks are unstable to train; pre-norm keeps a clean residual path
  from input to output.

  RoPE rather than learned position embeddings — position is applied by
  rotating the query and key vectors rather than adding a learned vector. It
  encodes relative distance, which is what attention actually needs.

  SwiGLU feed-forward — a gate multiplied by a value. Costs a third matrix and
  is worth it.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class Config:
    vocab_size: int = 16384
    n_layers: int = 8
    n_heads: int = 8
    n_embd: int = 384
    block_size: int = 256
    rope_base: int = 10000

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in fields})

    def __post_init__(self):
        if self.n_embd % self.n_heads:
            raise ValueError(
                f"n_embd {self.n_embd} must divide by n_heads {self.n_heads}"
            )


class Attention(nn.Module):
    """Masked multi-head self-attention.

    Every position looks at every earlier position and decides how much each
    matters. The mask is what makes this a language model rather than an
    autoencoder: without it, predicting the next token would be trivial
    because the model could simply read it.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.n_embd // cfg.n_heads
        self.scale = self.head_dim ** -0.5

        # One projection for all three, split afterwards. A single larger
        # matmul beats three smaller ones on the GPU.
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=cfg.rope_base)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)

        # (B, T, C) -> (B, heads, T, head_dim). Heads move to their own axis so
        # each attends independently.
        def heads(t):
            return t.reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        q, k, v = heads(q), heads(k), heads(v)

        # Position enters here, as a rotation of q and k — never of v. Values
        # carry content; only the matching should care where things are.
        q = self.rope(q)
        k = self.rope(k)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.proj(out)


class FeedForward(nn.Module):
    """SwiGLU. Two paths from the same input: one gated by SiLU, one plain,
    multiplied together. The gate lets the layer suppress its own output
    per-dimension, which a plain MLP cannot do."""

    def __init__(self, cfg: Config):
        super().__init__()
        # 8/3 rather than 4 keeps the parameter count level with a standard
        # MLP despite the third matrix — the convention Llama established.
        hidden = int(8 * cfg.n_embd / 3)
        hidden = 64 * ((hidden + 63) // 64)  # round up for GPU alignment

        self.gate = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.up = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.n_embd, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    """One layer: attention, then feed-forward, each normalised before and
    added back after.

    The residual additions are what make depth trainable. They give gradients
    an unobstructed path from the loss back to the first layer, so an eight
    layer stack learns rather than starving at the bottom.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.norm2 = nn.RMSNorm(cfg.n_embd)
        self.ffn = FeedForward(cfg)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


class Silbe(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = [Block(cfg) for _ in range(cfg.n_layers)]
        self.norm = nn.RMSNorm(cfg.n_embd)
        # No separate output matrix: the embedding is reused, transposed, to
        # score tokens. Weight tying saves vocab × n_embd parameters — at 16k
        # vocab and 384 width that is 6.3M, which would otherwise be 40% of
        # this model spent twice on the same information.

    def __call__(self, idx: mx.array) -> mx.array:
        x = self.embed(idx)

        # Causal mask, built once per forward pass. Position i may attend to
        # every j <= i and nothing after.
        T = idx.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T).astype(x.dtype)

        for block in self.blocks:
            x = block(x, mask)

        x = self.norm(x)
        return self.embed.as_linear(x)

    def loss(self, idx: mx.array, targets: mx.array) -> mx.array:
        """Cross-entropy between predictions and the true next tokens.

        Cross-entropy is the negative log of the probability the model gave to
        the correct answer. Certain and right scores near 0; certain and wrong
        scores enormously. That asymmetry is what teaches it to hedge when it
        does not know — which is most of what early training is.
        """
        logits = self(idx).astype(mx.float32)
        return nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="mean"
        )

    @property
    def n_params(self) -> int:
        from mlx.utils import tree_flatten

        return sum(v.size for _, v in tree_flatten(self.parameters()))
