"""Short benchmarks: speed/memory of a few real training steps, the max-batch search,
and depth / sequence-length scaling. Nothing here needs the dataset."""
from __future__ import annotations

import copy
import gc
import statistics
import time

import torch

from .config import ModelConfig, TrainConfig
from .model import GPT
from .train import resolve_amp, autocast_ctx, make_optimizer, OOM_ERRORS


def bench_steps(mc: ModelConfig, B: int, device="cuda", amp="auto", n_steps=5, seed=0):
    """Run n_steps real training steps (fwd + bwd + AdamW) on random tokens.
    Returns dict(ok, peak_mib, tokens_per_sec). ok=False means out-of-memory."""
    cuda = device == "cuda"
    dtype, use_scaler = resolve_amp(device, amp)
    tc = TrainConfig()
    model = opt = x = y = loss = None
    try:
        torch.manual_seed(seed)
        model = GPT(mc).to(device)
        opt = make_optimizer(model, tc, 1e-4, device)
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        x = torch.randint(0, mc.vocab_size, (B, mc.ctx), device=device)
        y = torch.randint(0, mc.vocab_size, (B, mc.ctx), device=device)
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(n_steps):
            t0 = time.perf_counter()
            with autocast_ctx(device, dtype):
                loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            loss.item()
            times.append(time.perf_counter() - t0)
        peak = torch.cuda.max_memory_allocated() / 2**20 if cuda else None
        t = statistics.median(times[2:] if len(times) > 3 else times)
        return dict(ok=True, peak_mib=peak, tokens_per_sec=B * mc.ctx / t, B=B)
    except OOM_ERRORS:
        return dict(ok=False, peak_mib=None, tokens_per_sec=None, B=B)
    finally:
        del model, opt, x, y, loss
        gc.collect()
        if cuda:
            torch.cuda.empty_cache()


def find_max_batch(trial, start=8, limit=4096, multiple=8, verbose=True):
    """Largest batch size for which `trial(B)` returns ok=True.
    `trial` -> dict(ok=bool, peak_mib=...).  Doubles until failure, then bisects
    (to a multiple of `multiple`).  Returns (best_B, log)."""
    log = []

    def t(B):
        r = trial(B)
        log.append(dict(B=B, ok=r["ok"], peak_mib=r.get("peak_mib"),
                        tokens_per_sec=r.get("tokens_per_sec")))
        if verbose:
            pk = r.get("peak_mib")
            print(f"    B={B:5d}  {'fits' if r['ok'] else 'OOM '}"
                  + (f"  peak {pk:8.0f} MiB" if pk else ""))
        return r["ok"]

    lo, hi, B = 0, None, start
    while B <= limit:
        if t(B):
            lo, B = B, B * 2
        else:
            hi = B
            break
    if lo == 0:
        return None, log
    if hi is None:                       # never failed up to `limit`
        return lo, log
    while hi - lo > multiple:
        mid = (lo + hi) // 2 // multiple * multiple
        if mid <= lo:
            break
        if t(mid):
            lo = mid
        else:
            hi = mid
    return lo, log


def max_batch_for(mc: ModelConfig, device="cuda", amp="auto", start=8, limit=4096):
    return find_max_batch(lambda B: bench_steps(mc, B, device, amp, n_steps=3), start, limit)


def scaling_table(base: ModelConfig, variants, device, amp, B, key, values):
    """Peak memory / speed of `variants` (list of dict overrides) while `key` (n_layer or ctx)
    takes each of `values`, at a fixed batch size."""
    rows = []
    for v in values:
        for name, over in variants:
            mc = copy.deepcopy(base)
            for k, val in over.items():
                setattr(mc, k, val)
            setattr(mc, key, v)
            r = bench_steps(mc, B, device, amp, n_steps=4)
            rows.append(dict(name=name, **{key: v}, B=B, ok=r["ok"], peak_mib=r["peak_mib"],
                             tokens_per_sec=r["tokens_per_sec"],
                             n_params=GPT(mc).num_params() if key == "n_layer" else None))
            pk = f"peak {r['peak_mib']:.0f} MiB  " if r["peak_mib"] else ""
            print(f"    {name:12s} {key}={v:<5} " + (pk + f"{r['tokens_per_sec']:.0f} tok/s" if r["ok"] else "OOM"))
    return rows
