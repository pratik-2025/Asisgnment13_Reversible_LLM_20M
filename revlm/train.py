"""Training loop that reports the numbers the assignment asks for:
final loss, tokens/s, peak GPU memory (+ curves, steps, time)."""
from __future__ import annotations

import json
import math
import statistics
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from .config import ModelConfig, TrainConfig
from .model import GPT

OOM_ERRORS = (torch.cuda.OutOfMemoryError,)


# ----------------------------------------------------------------------------
def resolve_amp(device: str, amp: str = "auto"):
    """-> (autocast dtype or None, use GradScaler). T4 has no fast bf16, so it gets fp16 + scaler."""
    if device != "cuda":
        return (torch.bfloat16, False) if amp == "bf16" else (None, False)
    if amp == "auto":
        major, _ = torch.cuda.get_device_capability()
        amp = "bf16" if major >= 8 else "fp16"
    if amp == "bf16":
        return torch.bfloat16, False
    if amp == "fp16":
        return torch.float16, True
    return None, False


def autocast_ctx(device, dtype):
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device, dtype=dtype)


def make_optimizer(model, tc: TrainConfig, lr, device):
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": tc.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    kw = {"fused": True} if device == "cuda" else {}
    return torch.optim.AdamW(groups, lr=lr, betas=tc.betas, **kw)


def peak_lr(tc: TrainConfig):
    if tc.lr_scaling == "sqrt":
        return min(tc.base_lr * math.sqrt(tc.batch_size / tc.ref_batch), tc.lr_cap)
    return tc.base_lr


def lr_at(step, total, lr_max, tc: TrainConfig):
    warm = max(10, int(tc.warmup_frac * total))
    if step < warm:
        return lr_max * (step + 1) / warm
    t = (step - warm) / max(1, total - warm)
    return lr_max * (tc.min_lr_frac + (1 - tc.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * t)))


@torch.no_grad()
def evaluate(model, data, device, dtype, tc: TrainConfig):
    model.eval()
    tot, n = 0.0, 0
    for x, y in data.val_batches(tc.eval_batch, tc.eval_tokens, device):
        with autocast_ctx(device, dtype):
            loss = model(x, y)
        tot += loss.item() * y.numel()
        n += y.numel()
    model.train()
    return tot / max(n, 1)


def _mb(x):
    return x / 2**20


# ----------------------------------------------------------------------------
def train(mc: ModelConfig, tc: TrainConfig, data, device="cuda", out_dir="results", verbose=True):
    """Train one run and write results/<run_name>.json. Returns the result dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cuda = device == "cuda"
    dtype, use_scaler = resolve_amp(device, tc.amp)

    torch.manual_seed(tc.seed)
    model = GPT(mc).to(device)
    model.train()
    n_params = model.num_params()
    lr_max = peak_lr(tc)
    opt = make_optimizer(model, tc, lr_max, device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    B, T = tc.batch_size, mc.ctx
    steps = min(tc.total_tokens // (B * T), data.n_seq // B)
    tokens_per_step = B * T
    eval_every = max(1, tc.eval_every_tokens // tokens_per_step)

    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    if verbose:
        print(f"[{tc.run_name}] mode={mc.mode} h={mc.h} B={B} steps={steps} "
              f"params={n_params/1e6:.2f}M amp={dtype} lr_peak={lr_max:.2e}")

    losses, step_times, val_curve = [], [], []
    peak_alloc = 0.0
    diverged = False
    t_start = time.time()
    for step in range(steps):
        t0 = time.perf_counter()
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, lr_max, tc)
        x, y = data.batch(step, B, device)
        with autocast_ctx(device, dtype):
            loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        scaler.step(opt)
        scaler.update()
        lv = loss.item()                       # also synchronises the GPU
        dt = time.perf_counter() - t0
        losses.append(lv)
        step_times.append(dt)
        if cuda:
            peak_alloc = max(peak_alloc, torch.cuda.max_memory_allocated())
        if not math.isfinite(lv):
            diverged = True
            print(f"[{tc.run_name}] loss is {lv} at step {step} -> stopping (diverged)")
            break
        if verbose and (step % tc.log_every == 0 or step == steps - 1):
            recent = statistics.mean(losses[-tc.log_every:])
            print(f"  step {step+1:5d}/{steps} tokens {(step+1)*tokens_per_step/1e6:6.2f}M "
                  f"loss {recent:.4f} {tokens_per_step/dt:9.0f} tok/s "
                  f"mem {_mb(peak_alloc):7.0f} MiB")
        if (step + 1) % eval_every == 0 or step == steps - 1:
            vl = evaluate(model, data, device, dtype, tc)
            val_curve.append(((step + 1) * tokens_per_step, vl))
            if verbose:
                print(f"  >> val loss {vl:.4f} at {(step+1)*tokens_per_step/1e6:.1f}M tokens")
            if cuda:
                torch.cuda.reset_peak_memory_stats()   # don't let eval memory leak into the train peak
    wall = time.time() - t_start

    fast = step_times[tc.warmup_steps_for_speed:] or step_times
    res = dict(
        run_name=tc.run_name,
        model=mc.to_dict(), train=tc.to_dict(),
        n_params=n_params,
        device=torch.cuda.get_device_name() if cuda else "cpu",
        autocast=str(dtype), fp16_grad_scaler=use_scaler,
        batch_size=B, ctx=T, tokens_per_step=tokens_per_step,
        steps_done=len(losses), steps_planned=steps,
        tokens_seen=len(losses) * tokens_per_step,
        lr_peak=lr_max,
        diverged=diverged,
        final_train_loss=statistics.mean(losses[-50:]) if losses else float("nan"),
        final_val_loss=val_curve[-1][1] if val_curve else float("nan"),
        tokens_per_sec_mean=tokens_per_step / statistics.mean(fast),
        tokens_per_sec_median=tokens_per_step / statistics.median(fast),
        step_time_ms_median=1000 * statistics.median(fast),
        peak_mem_allocated_mib=_mb(peak_alloc) if cuda else None,
        peak_mem_reserved_mib=_mb(torch.cuda.max_memory_reserved()) if cuda else None,
        wall_time_s=wall,
        train_losses=losses, val_curve=val_curve,
    )
    (out_dir / f"{tc.run_name}.json").write_text(json.dumps(res))
    if verbose:
        print(f"[{tc.run_name}] done: val {res['final_val_loss']:.4f}  train {res['final_train_loss']:.4f}  "
              f"{res['tokens_per_sec_median']:.0f} tok/s  peak {res['peak_mem_allocated_mib']} MiB  "
              f"{wall/60:.1f} min")
    del model, opt
    if cuda:
        torch.cuda.empty_cache()
    return res
