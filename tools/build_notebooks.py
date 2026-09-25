"""Generates the three notebooks in ../notebooks (so their code stays in sync with the package).
Run:  python tools/build_notebooks.py"""
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parent.parent / "notebooks"


def md(s):
    return nbf.v4.new_markdown_cell(s.strip("\n"))


def code(s):
    return nbf.v4.new_code_cell(s.strip("\n"))


SETUP = '''
SMOKE = False   # True = tiny CPU version of this whole notebook (used to test the code; not for results)
REPO_URL = "https://github.com/YOUR-USERNAME/reversible-llm-20m"   # <- edit: only needed on Colab
USE_DRIVE = False   # True = keep data + results on Google Drive so they survive a Colab disconnect

import os, sys, copy, json, subprocess
if os.path.exists("../revlm"):
    os.chdir("..")                                   # opened from the notebooks/ folder
elif not os.path.exists("revlm"):
    assert "YOUR-USERNAME" not in REPO_URL, "edit REPO_URL (top of this cell) to point at your GitHub repo"
    subprocess.run(["git", "clone", REPO_URL, "repo"], check=True)   # fresh Colab VM
    os.chdir("repo")
sys.path.insert(0, os.getcwd())
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "tokenizers", "datasets"], check=False)

import torch
from IPython.display import Image, display
from revlm import selftest, experiment as ex
from revlm.bench import bench_steps, scaling_table
from revlm.data import prepare_data, Data
from revlm.report import write_summary, plot_scaling

S = ex.settings(SMOKE)
DEV = ex.device()
RESULTS = "results_smoke" if SMOKE else "results"
DATA_DIR = "data_smoke" if SMOKE else "data"
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    RESULTS = "/content/drive/MyDrive/reversible-llm-20m/" + RESULTS
    DATA_DIR = "/content/drive/MyDrive/reversible-llm-20m/" + DATA_DIR
print("device:", DEV, torch.cuda.get_device_name() if DEV == "cuda" else "(no GPU!)" if not SMOKE else "(smoke test)")
if DEV != "cuda" and not SMOKE:
    print("WARNING: no GPU. In Colab use Runtime > Change runtime type > GPU.")
'''

SELFTEST = '''
# 30-second correctness check on THIS machine/GPU before spending an hour of compute.
assert selftest.main(DEV), "self-test failed - do not trust any numbers below"
'''

DATA = '''
# Downloads TinyStories, trains an 8k BPE tokenizer, writes exactly 50M training tokens (~3-6 min, cached).
prepare_data(DATA_DIR, vocab_size=8192, **S["data"])
data = Data(DATA_DIR, S["model"].ctx, seed=1337)
print(f"train tokens available: {len(data.train):,}   validation tokens: {len(data.val):,}   sequences: {data.n_seq:,}")
'''

MK = '''
def mk(mode, **over):
    """20M-parameter model config in the given mode (same weights/shape for every mode)."""
    mc = copy.deepcopy(S["model"]); mc.mode = mode
    for k, v in over.items():
        setattr(mc, k, v)
    return mc

from revlm.model import GPT
print(f"parameters: {GPT(mk('baseline')).num_params()/1e6:.2f}M   (layers={S['model'].n_layer}, d_model={S['model'].d_model}, vocab={S['model'].vocab_size}, ctx={S['model'].ctx})")
'''


def nb(cells, name):
    n = nbf.v4.new_notebook()
    n.cells = cells
    n.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbf.write(n, OUT / name)


# ------------------------------------------------------------------ 01
nb([
    md("""
# 01 - Baseline: 20M-parameter GPT, 50M tokens, biggest batch that fits

**Goal (assignment part 1).** Train a ~20M-parameter LLM on 50M tokens with an ordinary
(non-reversible) transformer. Find the biggest batch size that fits in GPU memory, fix it, and
record final loss, speed (tokens/s) and peak memory. Notebooks 02 and 03 reuse this batch size.

Set the runtime to a **GPU** (Runtime > Change runtime type). Edit `REPO_URL` in the next cell.
"""),
    code(SETUP), md("## 0. Sanity check"), code(SELFTEST), md("## 1. Data"), code(DATA), code(MK),
    md("""
## 2. Find the biggest batch that fits (baseline)
Doubles the batch until the GPU runs out of memory, then bisects. Each trial is a real training
step (forward, backward, AdamW update), so optimizer memory is included.
"""),
    code('''
srch = ex.search_or_load("maxbatch_baseline", mk("baseline"), RESULTS, S)
B_FIXED = srch["max_batch"]
print("baseline max batch:", B_FIXED)
'''),
    md("## 3. Train the baseline for 50M tokens at that batch size"),
    code('''
r0 = ex.run_or_load(mk("baseline"), ex.make_train_cfg(S, "01_baseline", B_FIXED), data, RESULTS)
B_FIXED = r0["batch_size"]          # (smaller only if a retry after OOM was needed)
ex.save(RESULTS, "fixed_batch", {"B_fixed": B_FIXED})
print(f"final val loss {r0['final_val_loss']:.4f} | {r0['tokens_per_sec_median']:,.0f} tokens/s | "
      f"peak {r0['peak_mem_allocated_mib']} MiB | {r0['wall_time_s']/60:.1f} min")
'''),
    md("## 4. Report"),
    code('''
write_summary(RESULTS)
display(Image(f"{RESULTS}/loss_curves.png"))
'''),
], "01_baseline.ipynb")

# ------------------------------------------------------------------ 02
nb([
    md("""
# 02 - Reversible training at the SAME batch size: which variant works best?

**Goal (assignment part 2).** Re-train with a *reversible* transformer at the batch size fixed in
notebook 01, and compare the reversible update rules:

| variant | update rule | note |
|---|---|---|
| `midpoint` | `x[l+1] = x[l-1] + 2h f(x[l])` | 2-step rule from the paper |
| `leapfrog` | `x[l+1] = 2x[l] - x[l-1] + h^2 f(x[l])` | 2nd-order, wave-equation style |
| `hamiltonian` | `q' = q + Attn(p);  p' = p + MLP(q')` | symplectic-Euler ("Euler-style") |

All variants have exactly the same parameters as the baseline. Report which one wins on loss.
"""),
    code(SETUP), md("## 0. Sanity check"), code(SELFTEST), md("## 1. Data"), code(DATA), code(MK),
    code('''
saved = ex.load(RESULTS, "fixed_batch")
B_FIXED = saved["B_fixed"] if saved else None      # <- if you skipped notebook 01, type the batch size here
assert B_FIXED, "run notebook 01 first (or set B_FIXED by hand)"
print("using batch size", B_FIXED)
'''),
    md("""
## 2. Quick speed / memory check at the fixed batch (5 real training steps each)
"""),
    code('''
bench = {}
for mode in ["baseline", "midpoint", "leapfrog", "hamiltonian"]:
    r = bench_steps(mk(mode), B_FIXED, DEV, "auto", n_steps=6)
    bench[mode] = r
    print(f"{mode:12s} ok={r['ok']}  peak={r['peak_mib']}  tok/s={r['tokens_per_sec']}")
ex.save(RESULTS, "bench_fixed_batch", bench)
'''),
    md("## 3. Train all three reversible variants for 50M tokens"),
    code('''
variants = {}
for mode in ["midpoint", "leapfrog", "hamiltonian"]:
    variants[mode] = ex.run_or_load(mk(mode), ex.make_train_cfg(S, f"02_{mode}", B_FIXED), data, RESULTS)
'''),
    md("""
## 4. Optional: does the step size `h` matter? (short pilot, off by default)
`h` is the step size of midpoint / leapfrog. The default `h = 0.5` makes midpoint's step `2h = 1`,
the same size as the baseline's residual step. Turn this on to test 0.25 / 0.5 / 1.0 on 5M tokens.
"""),
    code('''
RUN_H_SWEEP = False
if RUN_H_SWEEP:
    S_pilot = dict(S, total_tokens=5_000_000 if not SMOKE else 200_000, eval_every_tokens=2_500_000 if not SMOKE else 100_000)
    for mode in ["midpoint", "leapfrog"]:
        for h in [0.25, 0.5, 1.0]:
            tc = ex.make_train_cfg(S_pilot, f"hsweep_{mode}_h{h}", B_FIXED)
            r = ex.run_or_load(mk(mode, h=h), tc, data, RESULTS + "/hsweep")
            print(mode, h, "val loss", round(r["final_val_loss"], 4), "diverged" if r["diverged"] else "")
'''),
    md("## 5. Which variant wins?"),
    code('''
base = ex.load(RESULTS, "01_baseline")
ok = {m: r for m, r in variants.items() if not r["diverged"] and r["final_val_loss"] == r["final_val_loss"]}
winner = min(ok, key=lambda m: ok[m]["final_val_loss"])
ex.save(RESULTS, "winner", {"winner": winner, "h": ok[winner]["model"]["h"]})
print(f"{'run':14s}{'val loss':>10s}{'tok/s':>10s}{'peak MiB':>10s}")
for name, r in [("baseline", base)] + list(variants.items()):
    if r: print(f"{name:14s}{r['final_val_loss']:10.4f}{r['tokens_per_sec_median']:10,.0f}{(r['peak_mem_allocated_mib'] or 0):10,.0f}")
print("winner (lowest validation loss):", winner)
'''),
    md("""
## 6. Does memory really stop growing with depth? (baseline vs the winner)
Fixed batch, more and more layers, then longer sequences. Reversible memory should stay almost flat
in depth; the baseline's should grow linearly.
"""),
    code('''
variants_cfg = [("baseline", {"mode": "baseline"}), (winner, {"mode": winner})]
print("depth scaling"); rows_d = scaling_table(S["model"], variants_cfg, DEV, "auto", S["bench_batch"], "n_layer", S["depths"])
print("sequence-length scaling"); rows_c = scaling_table(S["model"], variants_cfg, DEV, "auto", max(1, S["bench_batch"] // 2), "ctx", S["ctxs"])
ex.save(RESULTS, "depth_scaling", {"rows": rows_d}); ex.save(RESULTS, "ctx_scaling", {"rows": rows_c})
if DEV == "cuda":
    plot_scaling(rows_d, "n_layer", f"{RESULTS}/depth_scaling.png", "number of transformer blocks")
    plot_scaling(rows_c, "ctx", f"{RESULTS}/ctx_scaling.png", "sequence length")
    display(Image(f"{RESULTS}/depth_scaling.png")); display(Image(f"{RESULTS}/ctx_scaling.png"))
'''),
    md("## 7. Loss curves and summary table"),
    code('''
write_summary(RESULTS)
display(Image(f"{RESULTS}/loss_curves.png")); display(Image(f"{RESULTS}/speed_memory.png"))
'''),
], "02_reversible_variants.ipynb")

# ------------------------------------------------------------------ 03
nb([
    md("""
# 03 - Reversible training pushed to the MAXIMUM batch size

**Goal (assignment part 3).** Take the winning reversible variant from notebook 02 and see how big a
batch a single GPU can hold now that activations are no longer stored. Train for 50M tokens at that
batch and compare with notebook 01.

Two things to watch for:
1. Reversibility removes the per-layer activations, but the **output logits** (`tokens x vocab`) still
   need memory. With a small model they become the limit. An optional *chunked loss* removes that limit.
2. A giant batch means **few optimizer steps** for the same 50M tokens. We scale the learning rate by
   `sqrt(batch/64)` and also run an un-scaled control so you can see the effect.
"""),
    code(SETUP), md("## 0. Sanity check"), code(SELFTEST), md("## 1. Data"), code(DATA), code(MK),
    code('''
B_FIXED = ex.load(RESULTS, "fixed_batch")["B_fixed"]
win = ex.load(RESULTS, "winner")
WINNER, H = win["winner"], win["h"]
print(f"winner: {WINNER} (h={H}); fixed batch from notebook 01: {B_FIXED}")
'''),
    md("## 2. How large a batch fits?"),
    code('''
s_plain = ex.search_or_load("maxbatch_rev_plainloss", mk(WINNER, h=H), RESULTS, S)
s_chunk = ex.search_or_load("maxbatch_rev_chunkedloss", mk(WINNER, h=H, loss_chunk=S["loss_chunk"]), RESULTS, S)
print("max batch, reversible:                ", s_plain["max_batch"])
print("max batch, reversible + chunked loss: ", s_chunk["max_batch"])
'''),
    md("## 3. Speed and memory as the batch grows (reversible vs baseline where the baseline still fits)"),
    code('''
ladder = sorted({B_FIXED, *[B_FIXED * k for k in (2, 3, 4, 6, 8) if B_FIXED * k <= s_chunk["max_batch"]],
                 s_plain["max_batch"], s_chunk["max_batch"]})
rows = []
for B in ladder:
    for name, mc in [("baseline", mk("baseline")), (WINNER, mk(WINNER, h=H)),
                     (WINNER + "+chunked_loss", mk(WINNER, h=H, loss_chunk=S["loss_chunk"]))]:
        r = bench_steps(mc, B, DEV, "auto", n_steps=4)
        rows.append(dict(name=name, B=B, **{k: r[k] for k in ("ok", "peak_mib", "tokens_per_sec")}))
        pk = f"{r['peak_mib']:8.0f}" if r["peak_mib"] else "     n/a"
        print(f"B={B:5d} {name:22s}", f"peak {pk} MiB  {r['tokens_per_sec']:9.0f} tok/s" if r["ok"] else "OOM")
ex.save(RESULTS, "batch_ladder", {"rows": rows})
'''),
    md("## 4. Train at the maximum batch size"),
    code('''
use_chunk = s_chunk["max_batch"] > s_plain["max_batch"]
B_MAX = max(s_chunk["max_batch"], s_plain["max_batch"])
mc_max = mk(WINNER, h=H, loss_chunk=S["loss_chunk"] if use_chunk else 0)
print(f"training at B={B_MAX}, chunked loss = {use_chunk}")
r3 = ex.run_or_load(mc_max, ex.make_train_cfg(S, "03_rev_maxbatch", B_MAX), data, RESULTS)
'''),
    md("Control: the same run with the learning rate NOT scaled up for the bigger batch."),
    code('''
RUN_LR_CONTROL = True
if RUN_LR_CONTROL:
    r3b = ex.run_or_load(mc_max, ex.make_train_cfg(S, "03_rev_maxbatch_lr_unscaled", r3["batch_size"], lr_scaling="none"), data, RESULTS)
'''),
    md("""
## 5. Final report
`PRICE_PER_HOUR` is **your assumption** for what one hour of this GPU costs on a cloud provider
(Colab free = $0). It only feeds the last column - edit it before quoting cost numbers.
"""),
    code('''
PRICE_PER_HOUR = 0.35     # assumption - edit
write_summary(RESULTS, price_per_hour=PRICE_PER_HOUR)
display(Image(f"{RESULTS}/loss_curves.png")); display(Image(f"{RESULTS}/speed_memory.png"))
print(open(f"{RESULTS}/summary.md").read())
'''),
], "03_reversible_max_batch.ipynb")

print("wrote notebooks to", OUT)
