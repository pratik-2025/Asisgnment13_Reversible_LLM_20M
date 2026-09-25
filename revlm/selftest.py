"""Correctness checks. Run them on a laptop CPU *and* at the top of each Colab notebook
(python -m revlm.selftest) so a bug shows up in 30 seconds, not after a 20-minute run.

What is checked
  1. Same parameter count and identical initial weights for all four modes.
  2. Reversibility: the input state rebuilt from the output state matches the true input.
  3. Gradients from the memory-saving backward == gradients from ordinary autograd.
  4. Chunked loss == normal loss (value and gradients).
  5. 'Bytes saved for backward' really is (almost) flat in depth for reversible modes.
"""
from __future__ import annotations

import copy
import sys

import torch

from .config import ModelConfig, MODES
from .model import GPT
from .reversible import step_forward, step_backward


def _tiny(mode, n_layer=4, **kw):
    return ModelConfig(vocab_size=101, n_layer=n_layer, n_head=2, d_model=32, ctx=24,
                       mode=mode, **kw)


def _batch(cfg, B=3, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, cfg.vocab_size, (B, cfg.ctx), generator=g).to(device)
    y = torch.randint(0, cfg.vocab_size, (B, cfg.ctx), generator=g).to(device)
    return x, y


def check_same_init():
    counts, sd = {}, {}
    for m in MODES:
        torch.manual_seed(0)
        net = GPT(_tiny(m))
        counts[m] = net.num_params()
        sd[m] = net.state_dict()
    assert len(set(counts.values())) == 1, counts
    for m in MODES[1:]:
        for k in sd["baseline"]:
            assert torch.equal(sd["baseline"][k], sd[m][k]), (m, k)
    return counts["baseline"]


def check_reconstruction(dtype=torch.float64, device="cpu"):
    """forward one step, then invert it; must land back on the input pair."""
    errs = {}
    for m in ("midpoint", "leapfrog", "hamiltonian"):
        torch.manual_seed(1)
        net = GPT(_tiny(m, n_layer=2, h=0.5)).to(device, dtype)
        blk = net.blocks[0]
        a = torch.randn(2, 24, 32, dtype=dtype, device=device)
        b = torch.randn(2, 24, 32, dtype=dtype, device=device)
        with torch.no_grad():
            u, v = step_forward(blk, a, b, m, 0.5)
        ga, gb = torch.randn_like(a), torch.randn_like(b)
        a2, b2, _, _ = step_backward(blk, u, v, ga, gb, m, 0.5)
        blk.zero_grad()
        errs[m] = max((a - a2).abs().max().item(), (b - b2).abs().max().item())
    return errs


def check_gradients(dtype=torch.float64, device="cpu", n_layer=6, loss_chunk=0):
    """memory-saving backward must give the same gradients as plain autograd."""
    errs = {}
    for m in ("midpoint", "leapfrog", "hamiltonian"):
        grads = {}
        for impl in ("autograd", "memory_saving"):
            torch.manual_seed(2)
            cfg = _tiny(m, n_layer=n_layer, h=0.5, rev_impl=impl, loss_chunk=loss_chunk)
            net = GPT(cfg).to(device, dtype)
            x, y = _batch(cfg, device=device)
            loss = net(x, y)
            loss.backward()
            grads[impl] = (loss.item(), {k: p.grad.clone() for k, p in net.named_parameters()})
        l0, g0 = grads["autograd"]
        l1, g1 = grads["memory_saving"]
        worst = max(((g0[k] - g1[k]).abs().max() / (g0[k].abs().max() + 1e-30)).item() for k in g0)
        errs[m] = (abs(l0 - l1), worst)
    return errs


def check_chunked_loss(dtype=torch.float64, device="cpu"):
    torch.manual_seed(3)
    out = {}
    for m in ("baseline", "midpoint"):
        res = []
        for chunk in (0, 17):
            torch.manual_seed(3)
            net = GPT(_tiny(m, loss_chunk=chunk)).to(device, dtype)
            x, y = _batch(net.cfg, device=device)
            loss = net(x, y)
            loss.backward()
            res.append((loss.item(), net.tok.weight.grad.clone()))
        out[m] = (abs(res[0][0] - res[1][0]),
                  ((res[0][1] - res[1][1]).abs().max() / res[0][1].abs().max()).item())
    return out


def check_amp_gradients(device="cpu"):
    """Under mixed precision (fp16 on GPU, bf16 on CPU) the memory-saving gradients must
    still point the same way as ordinary autograd (cosine similarity ~ 1)."""
    dtype = torch.float16 if device == "cuda" else torch.bfloat16
    out = {}
    for m in ("midpoint", "leapfrog", "hamiltonian"):
        gs = {}
        for impl in ("autograd", "memory_saving"):
            torch.manual_seed(2)
            cfg = _tiny(m, n_layer=6, h=0.5, rev_impl=impl)
            net = GPT(cfg).to(device)
            x, y = _batch(cfg, device=device)
            with torch.autocast(device, dtype=dtype):
                loss = net(x, y)
            loss.backward()
            gs[impl] = torch.cat([p.grad.flatten().float() for p in net.parameters()])
        assert torch.isfinite(gs["memory_saving"]).all(), m
        out[m] = torch.nn.functional.cosine_similarity(gs["autograd"], gs["memory_saving"], dim=0).item()
    return out


def saved_bytes(net, x, y):
    """Total bytes of tensors autograd keeps for backward (hardware independent)."""
    total = [0]

    def pack(t):
        total[0] += t.numel() * t.element_size()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        loss = net(x, y)
    loss.backward()
    return total[0]


def check_saved_bytes(depths=(2, 4, 8, 16)):
    rows = {}
    for m in ("baseline", "midpoint"):
        rows[m] = []
        for L in depths:
            torch.manual_seed(0)
            net = GPT(_tiny(m, n_layer=L))
            x, y = _batch(net.cfg)
            rows[m].append(saved_bytes(net, x, y))
    return depths, rows


def main(device="cpu"):
    ok = True
    print(f"device={device}")
    print("1. params identical across modes:", check_same_init())
    r = check_reconstruction(device=device)
    print("2. reconstruction max |error| (float64):", {k: f"{v:.1e}" for k, v in r.items()})
    ok &= all(v < 1e-9 for v in r.values())
    g = check_gradients(device=device)
    print("3. memory-saving vs autograd (loss diff, worst rel. grad diff):",
          {k: (f"{a:.1e}", f"{b:.1e}") for k, (a, b) in g.items()})
    ok &= all(b < 1e-8 for _, b in g.values())
    c = check_chunked_loss(device=device)
    print("4. chunked vs normal loss (loss diff, rel. grad diff):",
          {k: (f"{a:.1e}", f"{b:.1e}") for k, (a, b) in c.items()})
    ok &= all(b < 1e-8 for _, b in c.values())
    a = check_amp_gradients(device)
    print("4b. mixed-precision grad cosine(memory-saving, autograd):", {k: f"{v:.5f}" for k, v in a.items()})
    ok &= all(v > 0.99 for v in a.values())
    if device == "cpu":
        depths, rows = check_saved_bytes()
        print("5. bytes saved for backward vs depth", depths)
        for m, v in rows.items():
            print(f"   {m:9s}", v)
        ok &= rows["midpoint"][-1] < rows["baseline"][-1]
        ok &= rows["midpoint"][-1] <= rows["midpoint"][0] * 1.05
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return ok


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sys.exit(0 if main(dev) else 1)
