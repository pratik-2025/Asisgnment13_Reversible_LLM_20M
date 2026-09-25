"""Glue used by the three notebooks: shared settings, resumable runs, OOM fallback."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from .config import ModelConfig, TrainConfig
from .train import train, OOM_ERRORS
from .bench import max_batch_for

# --------------------------------------------------------------------------
# Settings. FULL = the real assignment (20M params, 50M tokens). SMOKE = tiny CPU
# version so the notebooks can be executed anywhere in a couple of minutes.
# --------------------------------------------------------------------------


def settings(smoke: bool = False):
    if smoke:
        return dict(
            model=ModelConfig(vocab_size=8192, n_layer=6, n_head=2, d_model=64, ctx=128),
            data=dict(train_tokens=1_000_000, val_tokens=100_000, tok_stories=3000),
            total_tokens=300_000, eval_every_tokens=150_000, eval_tokens=50_000,
            search_start=4, search_limit=16, ref_batch=16, base_lr=2e-3, loss_chunk=1024,
            bench_batch=8, depths=[2, 4, 8], ctxs=[64, 128],
        )
    return dict(
        model=ModelConfig(vocab_size=8192, n_layer=14, n_head=5, d_model=320, ctx=512),
        data=dict(train_tokens=50_000_000, val_tokens=1_000_000, tok_stories=50_000),
        total_tokens=50_000_000, eval_every_tokens=5_000_000, eval_tokens=1_000_000,
        search_start=8, search_limit=4096, ref_batch=64, base_lr=1e-3, loss_chunk=8192,
        bench_batch=32, depths=[7, 14, 28, 56], ctxs=[256, 512, 1024, 2048],
    )


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_train_cfg(S, name, B, **kw):
    return TrainConfig(run_name=name, total_tokens=S["total_tokens"], batch_size=B,
                       ref_batch=S["ref_batch"], base_lr=S["base_lr"],
                       eval_every_tokens=S["eval_every_tokens"], eval_tokens=S["eval_tokens"], **kw)


def load(results_dir, name):
    p = Path(results_dir) / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def save(results_dir, name, obj):
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    (Path(results_dir) / f"{name}.json").write_text(json.dumps(obj, indent=1))


def run_or_load(mc: ModelConfig, tc: TrainConfig, data, results_dir, dev=None, fallback=True, multiple=8):
    """Train unless results/<name>.json already exists (so a crashed Colab session can resume).
    If a run hits out-of-memory (batch chosen right at the limit), retry with a smaller batch
    and record it in the JSON so the report is honest about it."""
    dev = dev or device()
    done = load(results_dir, tc.run_name)
    if done is not None:
        print(f"[{tc.run_name}] already done -> loaded from results/")
        return done
    tries = []
    while True:
        try:
            res = train(mc, tc, data, device=dev, out_dir=results_dir)
            if tries:
                res["oom_retries_batch_sizes"] = tries
                save(results_dir, tc.run_name, res)
            return res
        except OOM_ERRORS:
            if not fallback or tc.batch_size - multiple < multiple:
                raise
            print(f"[{tc.run_name}] OOM at B={tc.batch_size}; retrying with B={tc.batch_size - multiple}")
            tries.append(tc.batch_size)
            tc = copy.deepcopy(tc)
            tc.batch_size -= multiple
            torch.cuda.empty_cache()


def search_or_load(name, mc, results_dir, S, dev=None, amp="auto"):
    """Max-batch search, cached in results/<name>.json."""
    dev = dev or device()
    done = load(results_dir, name)
    if done is not None:
        print(f"[{name}] max batch = {done['max_batch']} (cached)")
        return done
    best, log = max_batch_for(mc, dev, amp, start=S["search_start"], limit=S["search_limit"])
    out = dict(name=name, max_batch=best, log=log, model=mc.to_dict())
    save(results_dir, name, out)
    print(f"[{name}] max batch = {best}")
    return out
