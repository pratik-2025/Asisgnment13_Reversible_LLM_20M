"""A ~20M-parameter GPT that can run in four 'residual' styles.

baseline    : ordinary pre-LN transformer  x <- x + f(x)   (this is forward Euler with step 1)
midpoint    : x[l+1] = x[l-1] + 2h f(x[l])                 (reversible, 2-step)
leapfrog    : x[l+1] = 2x[l] - x[l-1] + h^2 f(x[l])        (reversible, 2nd order)
hamiltonian : q' = q + Attn(p);  p' = p + MLP(q')          (reversible, symplectic Euler)

All four use EXACTLY the same parameters (same shapes, same initialisation for a
given seed), so differences in loss/speed/memory come from the update rule only.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig
from .reversible import run_stack


def _up(t):
    """Compute the loss in at least float32 (half-precision logits -> float32)."""
    return t.float() if t.dtype in (torch.float16, torch.bfloat16) else t


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.n_head = cfg.n_head
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, 4 * d, bias=False)
        self.fc2 = nn.Linear(4 * d, d, bias=False)

    # --- the two sub-layers -------------------------------------------------
    def attn_branch(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=-1)
        shp = (B, T, self.n_head, D // self.n_head)
        q, k, v = (t.view(shp).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, D))

    def mlp_branch(self, x):
        return self.fc2(F.gelu(self.fc1(self.ln2(x))))

    # --- residual function used by baseline / midpoint / leapfrog ------------
    def f(self, x):
        """block(x) - x : the whole transformer block written as a 'change' to x."""
        a = self.attn_branch(x)
        return a + self.mlp_branch(x + a)


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.ctx, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        # output head is tied to the token embedding (no extra parameters)
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("proj.weight") or n.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------
    def hidden(self, idx):
        cfg = self.cfg
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        if cfg.mode == "baseline":
            for blk in self.blocks:
                x = x + blk.f(x)
            out = x
        else:
            # both members of the starting pair are the embedding
            a, b = x, x
            ms = cfg.rev_impl == "memory_saving"
            a, b = run_stack(self.blocks, a, b, cfg.mode, cfg.h, ms)
            out = a if cfg.mode == "hamiltonian" else b   # p-stream / latest state
        return self.ln_f(out)

    def _logits(self, h):
        return F.linear(h, self.tok.weight)

    def forward(self, idx, targets=None):
        h = self.hidden(idx)
        if targets is None:
            return self._logits(h)
        cfg = self.cfg
        hf, tf = h.reshape(-1, h.size(-1)), targets.reshape(-1)
        if cfg.loss_chunk and cfg.loss_chunk < hf.size(0):
            total = hf.new_zeros((), dtype=torch.float32)
            w = self.tok.weight

            def piece(hc, tc, w):
                return F.cross_entropy(_up(F.linear(hc, w)), tc, reduction="sum")

            for hc, tc in zip(hf.split(cfg.loss_chunk), tf.split(cfg.loss_chunk)):
                # checkpoint => this chunk's logits are freed right away and
                # recomputed during backward; peak logit memory = one chunk.
                total = total + checkpoint(piece, hc, tc, w, use_reentrant=False)
            return total / tf.numel()
        return F.cross_entropy(_up(self._logits(hf)), tf)
