"""Plain dataclasses that describe a model and a training run."""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

MODES = ("baseline", "midpoint", "leapfrog", "hamiltonian")


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    n_layer: int = 14          # number of transformer blocks
    n_head: int = 5
    d_model: int = 320
    ctx: int = 512             # sequence length
    mode: str = "baseline"     # baseline | midpoint | leapfrog | hamiltonian
    h: float = 0.5             # step size for midpoint / leapfrog (ignored by baseline, hamiltonian)
    rev_impl: str = "memory_saving"   # memory_saving (custom backward) | autograd (stores everything)
    loss_chunk: int = 0        # 0 = normal loss; >0 = compute logits+loss in chunks of this many tokens

    def __post_init__(self):
        assert self.mode in MODES, f"unknown mode {self.mode}"
        assert self.d_model % self.n_head == 0
        assert self.rev_impl in ("memory_saving", "autograd")

    def to_dict(self):
        return asdict(self)


@dataclass
class TrainConfig:
    run_name: str = "run"
    total_tokens: int = 50_000_000
    batch_size: int = 32            # sequences per step (no gradient accumulation)
    base_lr: float = 1e-3           # LR used at ref_batch
    ref_batch: int = 64
    lr_scaling: str = "sqrt"        # sqrt | none
    lr_cap: float = 2e-3
    min_lr_frac: float = 0.1
    warmup_frac: float = 0.02
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    seed: int = 1337
    eval_every_tokens: int = 5_000_000
    eval_tokens: int = 1_000_000
    eval_batch: int = 16
    log_every: int = 20             # steps
    amp: str = "auto"               # auto | fp16 | bf16 | none
    warmup_steps_for_speed: int = 20

    def to_dict(self):
        return asdict(self)
