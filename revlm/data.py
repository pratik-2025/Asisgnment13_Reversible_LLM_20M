"""TinyStories -> 8k-vocab BPE -> one flat uint16 file of exactly N tokens.

Why our own tokenizer?  With GPT-2's 50k vocabulary the embedding table alone would
be ~16M of the 20M parameters, leaving almost nothing for the transformer layers.
An 8k vocabulary keeps the budget in the layers, which is where reversibility acts.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

EOT = "<|endoftext|>"


def _stories(split: str):
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    for ex in ds:
        t = ex["text"].strip()
        if t:
            yield t


def train_tokenizer(out_dir: Path, vocab_size=8192, n_stories=50_000):
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[EOT],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    def it():
        for i, s in enumerate(_stories("train")):
            if i >= n_stories:
                break
            yield s
    tok.train_from_iterator(it(), trainer)
    tok.save(str(out_dir / "tokenizer.json"))
    return tok


def _encode_to_file(tok, split, path: Path, n_tokens: int, batch=2000):
    eot = tok.token_to_id(EOT)
    buf = np.empty(n_tokens, dtype=np.uint16)
    n, texts = 0, []

    def flush(texts, n):
        for ids in (e.ids for e in tok.encode_batch(texts)):
            ids = ids + [eot]
            take = min(len(ids), n_tokens - n)
            buf[n:n + take] = ids[:take]
            n += take
            if n >= n_tokens:
                break
        return n

    for s in _stories(split):
        texts.append(s)
        if len(texts) >= batch:
            n = flush(texts, n)
            texts = []
            print(f"\r  {split}: {n:,}/{n_tokens:,} tokens", end="", flush=True)
            if n >= n_tokens:
                break
    if n < n_tokens and texts:
        n = flush(texts, n)
    print()
    if n < n_tokens:
        print(f"  WARNING: only {n:,} tokens available for {split}")
    buf[:n].tofile(path)
    return n


def prepare_data(data_dir="data", train_tokens=50_000_000, val_tokens=1_000_000,
                 vocab_size=8192, tok_stories=50_000):
    """Idempotent: skips work that is already on disk. Returns the data directory."""
    d = Path(data_dir)
    d.mkdir(parents=True, exist_ok=True)
    meta_p = d / "meta.json"
    want = dict(train_tokens=train_tokens, val_tokens=val_tokens, vocab_size=vocab_size)
    if meta_p.exists() and all(json.loads(meta_p.read_text()).get(k) == v for k, v in want.items()):
        return d
    from tokenizers import Tokenizer
    if (d / "tokenizer.json").exists():
        tok = Tokenizer.from_file(str(d / "tokenizer.json"))
    else:
        print("training tokenizer ...")
        tok = train_tokenizer(d, vocab_size, tok_stories)
    print("tokenising train ...")
    ntr = _encode_to_file(tok, "train", d / "train.bin", train_tokens)
    print("tokenising validation ...")
    nva = _encode_to_file(tok, "validation", d / "val.bin", val_tokens)
    meta_p.write_text(json.dumps(dict(want, train_tokens_actual=ntr, val_tokens_actual=nva)))
    return d


class Data:
    """Serves non-overlapping (ctx+1)-token windows in a fixed shuffled order, so a run
    sees each training token exactly once (one epoch = the token budget)."""

    def __init__(self, data_dir, ctx, seed=1337):
        d = Path(data_dir)
        self.meta = json.loads((d / "meta.json").read_text())
        self.train = np.memmap(d / "train.bin", dtype=np.uint16, mode="r")
        self.val = np.memmap(d / "val.bin", dtype=np.uint16, mode="r")
        self.ctx = ctx
        self.vocab_size = self.meta["vocab_size"]
        self.n_seq = (len(self.train) - 1) // ctx
        self.order = np.random.default_rng(seed).permutation(self.n_seq)

    def batch(self, step, B, device):
        import torch
        rows = self.order[step * B:(step + 1) * B]
        x = np.stack([self.train[r * self.ctx: r * self.ctx + self.ctx] for r in rows]).astype(np.int64)
        y = np.stack([self.train[r * self.ctx + 1: r * self.ctx + self.ctx + 1] for r in rows]).astype(np.int64)
        return torch.from_numpy(x).to(device, non_blocking=True), torch.from_numpy(y).to(device, non_blocking=True)

    def val_batches(self, B, n_tokens, device):
        import torch
        n_seq = min((len(self.val) - 1) // self.ctx, n_tokens // self.ctx)
        for s in range(0, n_seq, B):
            rows = range(s, min(s + B, n_seq))
            x = np.stack([self.val[r * self.ctx: r * self.ctx + self.ctx] for r in rows]).astype(np.int64)
            y = np.stack([self.val[r * self.ctx + 1: r * self.ctx + self.ctx + 1] for r in rows]).astype(np.int64)
            yield torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
