"""Memory-saving backward pass for reversible stacks.

Idea in one paragraph (plain English)
-------------------------------------
A normal network keeps every layer's intermediate results in memory so that the
backward pass can use them. A *reversible* network is built so that a layer's
input can be computed back from its output. So in the forward pass we throw all
the intermediates away and keep only the final state. In the backward pass we
walk from the last layer to the first; at each layer we (1) rebuild the layer's
input from its output, (2) re-run the layer once to get the gradients, and
(3) move on. Memory no longer grows with the number of layers; we pay with
extra compute (each layer's forward runs twice).

State is always a pair (a, b):
  * midpoint / leapfrog : (a, b) = (x[l-1], x[l])   -> new pair (x[l], x[l+1])
  * hamiltonian         : (a, b) = (p, q)           -> new pair (p', q')

Every scheme implements three functions on (state, block):
  forward_step(block, a, b, cfg)      -> (a2, b2)
  backward_step(block, a2, b2, ga2, gb2, cfg) -> (a, b, ga, gb)
     reconstructs the input pair from the output pair and returns the
     gradients w.r.t. the input pair. Parameter gradients are accumulated into
     `.grad` by an inner autograd call.
"""
from __future__ import annotations

import torch

# ----------------------------------------------------------------------------
# Forward steps (used by the plain-autograd path AND by the memory-saving path)
# ----------------------------------------------------------------------------


def step_forward(block, a, b, mode: str, h: float):
    if mode == "midpoint":
        # x[l+1] = x[l-1] + 2h f(x[l]);   pair (x[l-1], x[l]) -> (x[l], x[l+1])
        return b, a + (2.0 * h) * block.f(b)
    if mode == "leapfrog":
        # x[l+1] = 2 x[l] - x[l-1] + h^2 f(x[l])
        return b, 2.0 * b - a + (h * h) * block.f(b)
    if mode == "hamiltonian":
        # symplectic Euler:  q' = q + Attn(p);   p' = p + MLP(q')
        p, q = a, b
        q2 = q + block.attn_branch(p)
        p2 = p + block.mlp_branch(q2)
        return p2, q2
    raise ValueError(mode)


# ----------------------------------------------------------------------------
# Backward steps: invert + vector-Jacobian product
# ----------------------------------------------------------------------------


def _grad_through(fn, x, grad_out):
    """Run fn(x) with grad enabled, backprop `grad_out` through it.
    Returns (fn(x).detach(), grad wrt x). Parameter grads accumulate in .grad."""
    x = x.detach().requires_grad_(True)
    with torch.enable_grad():
        y = fn(x)
    torch.autograd.backward(y, grad_out.to(y.dtype))
    return y.detach(), x.grad


def step_backward(block, u, v, gu, gv, mode: str, h: float):
    """(u, v) is the OUTPUT pair of a step; (gu, gv) the gradients w.r.t. it.
    Returns the INPUT pair (a, b) and gradients (ga, gb) w.r.t. it."""
    if mode == "midpoint":
        # forward: (a, b) -> (u, v) = (b, a + 2h f(b))
        b = u
        fb, gjt = _grad_through(block.f, b, (2.0 * h) * gv)  # gjt = 2h J^T gv
        a = v - (2.0 * h) * fb
        return a, b, gv, gu + gjt
    if mode == "leapfrog":
        # forward: (a, b) -> (u, v) = (b, 2b - a + h^2 f(b))
        b = u
        fb, gjt = _grad_through(block.f, b, (h * h) * gv)
        a = 2.0 * b - v + (h * h) * fb
        return a, b, -gv, gu + 2.0 * gv + gjt
    if mode == "hamiltonian":
        # forward: p' = p + G(q'), q' = q + F(p)   with (u, v) = (p', q')
        p2, q2 = u, v
        g_mlp, gq_from_mlp = _grad_through(block.mlp_branch, q2, gu)
        p = p2 - g_mlp
        gq2 = gv + gq_from_mlp                      # q' feeds both outputs
        f_att, gp_from_att = _grad_through(block.attn_branch, p, gq2)
        q = q2 - f_att
        return p, q, gu + gp_from_att, gq2
    raise ValueError(mode)


# ----------------------------------------------------------------------------
# The autograd Function that ties it together
# ----------------------------------------------------------------------------


def _autocast_state(t: torch.Tensor):
    """(device type, autocast on?, autocast dtype) so backward can recompute in the same precision."""
    dev = t.device.type
    try:
        return dev, torch.is_autocast_enabled(dev), torch.get_autocast_dtype(dev)
    except TypeError:                      # torch < 2.4 (single-argument legacy API)
        if dev == "cuda":
            return dev, torch.is_autocast_enabled(), torch.get_autocast_gpu_dtype()
        return dev, torch.is_autocast_cpu_enabled(), torch.get_autocast_cpu_dtype()


class ReversibleStack(torch.autograd.Function):
    """Runs all blocks; saves ONLY the final state pair for backward."""

    @staticmethod
    def forward(ctx, a, b, blocks, mode, h):
        ctx.blocks, ctx.mode, ctx.h = blocks, mode, h
        ctx.ac = _autocast_state(a)
        # Function.forward already runs without autograd recording.
        for blk in blocks:
            a, b = step_forward(blk, a, b, mode, h)
        ctx.save_for_backward(a, b)      # <- the only tensors kept
        return a, b

    @staticmethod
    def backward(ctx, ga, gb):
        u, v = ctx.saved_tensors
        dev, enabled, dtype = ctx.ac
        blocks, mode, h = ctx.blocks, ctx.mode, ctx.h
        if ga is None:
            ga = torch.zeros_like(u)
        if gb is None:
            gb = torch.zeros_like(v)
        kw = {"dtype": dtype} if dtype is not None else {}
        with torch.autocast(device_type=dev, enabled=enabled, **kw):
            for blk in reversed(blocks):
                u, v, ga, gb = step_backward(blk, u, v, ga, gb, mode, h)
        return ga, gb, None, None, None


def run_stack(blocks, a, b, mode, h, memory_saving: bool):
    """Apply all blocks to the state pair (a, b)."""
    if memory_saving and torch.is_grad_enabled() and (a.requires_grad or b.requires_grad):
        return ReversibleStack.apply(a, b, list(blocks), mode, h)
    for blk in blocks:                    # plain autograd (also used for eval)
        a, b = step_forward(blk, a, b, mode, h)
    return a, b
