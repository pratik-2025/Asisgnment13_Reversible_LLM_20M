# Reversible training of a 20M-parameter LLM (ERA V5, Session 13)

> **Status of this README.** The code, the notebooks and the correctness checks are finished and tested.
> The *GPU results* (final loss, tokens/s, peak memory) are **not filled in yet** because they have to
> come from a real Colab run. Every cell marked `PENDING` is filled from `results/summary.md`, which the
> notebooks write themselves. Nothing below has been typed in by hand from memory.

## 1. The assignment

Train a ~20M-parameter LLM on 50M tokens, three times, and report final loss, speed (tokens/s),
peak memory and other findings:

1. **Baseline** - ordinary transformer, the biggest batch size that fits on my GPU.
2. **Reversible** - same batch size, try the reversible variants (midpoint, Euler-style, leapfrog) and say which one works best.
3. **Reversible, maximum batch** - keep the best variant and push the batch size as far as one GPU allows.

Source: ERA V5 Session 13 (distributed training part 2) and the paper it is based on,
*Reversing Large Language Models for Efficient Training and Fine-Tuning* ([arXiv:2512.02056](https://arxiv.org/abs/2512.02056)).

| Notebook | What it does |
|---|---|
| [`notebooks/01_baseline.ipynb`](notebooks/01_baseline.ipynb) | finds the max batch for the normal model, trains 50M tokens |
| [`notebooks/02_reversible_variants.ipynb`](notebooks/02_reversible_variants.ipynb) | same batch, three reversible variants, picks the winner, memory-vs-depth test |
| [`notebooks/03_reversible_max_batch.ipynb`](notebooks/03_reversible_max_batch.ipynb) | pushes the batch size to the limit, trains 50M tokens, final report |

## 2. Reversibility in plain English

Training a neural network has two passes. The **forward pass** computes the answer. The **backward
pass** works out how to improve each weight. The backward pass needs the *intermediate numbers*
(activations) that the forward pass produced, so a normal network keeps all of them in GPU memory.
More layers, longer text or bigger batches all mean more stored numbers, and that is usually what
runs the GPU out of memory - not the weights.

A **reversible** network is built so that each layer's input can be calculated back from its output.
So the forward pass keeps *only the last layer's output* and throws the rest away. During the backward
pass, for each layer going from last to first:

1. rebuild that layer's input from its output (this is the "reverse" part),
2. run the layer once more to get the gradients,
3. move to the layer before it.

**The trade:** memory for activations stops growing with the number of layers, but every layer's forward
computation is done twice, so training is slower per step (the paper and the session both say roughly
30-50% slower per step). The hope is that the freed memory allows a much bigger batch, which wins some
of the speed back.

**The price of the trick:** the layer function must be invertible by construction, and anything random
inside a layer (dropout) would have to be replayed exactly. I use no dropout.

## 3. The update rules I compare

Every block below is the *same* transformer block (attention + MLP, same weights). Only the way the
block's result is added to the running state changes. Let `f(x)` = "block(x) minus x", i.e. what the
block adds.

| name | forward rule | how to go backwards | reversible? |
|---|---|---|---|
| `baseline` (forward Euler) | `x[l+1] = x[l] + f(x[l])` | - (needs stored activations) | no |
| `midpoint` | `x[l+1] = x[l-1] + 2h * f(x[l])` | `x[l-1] = x[l+1] - 2h * f(x[l])` | yes |
| `leapfrog` | `x[l+1] = 2x[l] - x[l-1] + h^2 * f(x[l])` | `x[l-1] = 2x[l] - x[l+1] + h^2 * f(x[l])` | yes |
| `hamiltonian` (symplectic Euler) | `q' = q + Attn(p)` then `p' = p + MLP(q')` | `p = p' - MLP(q')` then `q = q' - Attn(p)` | yes |

- `midpoint` and `leapfrog` keep **two** states (the previous and the current layer's output). To go
  backwards you need the last two, then each step recovers one more.
- `hamiltonian` keeps two streams, `p` and `q`, and updates them one after the other. It is the
  "Euler-style" variant: the baseline is plain forward Euler, and this is its reversible sibling.
- `h` is a fixed step size. I use `h = 0.5`, so midpoint's `2h = 1` matches the baseline's step size.
  I did not tune it. Notebook 02 has an optional switch (`RUN_H_SWEEP`) that tries 0.25 / 0.5 / 1.0.

All four modes have **exactly the same parameters and the same starting weights** (checked by a test), so
any difference in loss, speed or memory comes from the update rule alone.

Choices that are mine, not the paper's (see section 8): both starting states are the embedding, the
readout is the latest state (`p` for hamiltonian), and there is no dropout.

## 4. Setup

| | |
|---|---|
| model | GPT-style, pre-LayerNorm, **14 blocks, d_model 320, 5 heads, MLP 4x, context 512**, tied input/output embedding, learned positions |
| parameters | **20,007,040** (counted by the code) |
| tokenizer | byte-level BPE, **8,192 tokens**, trained on 50k TinyStories stories |
| data | [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) train split, exactly **50,000,000 tokens**, each seen once (one epoch, fixed shuffled order). 1M tokens of the validation split for validation loss |
| optimizer | AdamW, betas (0.9, 0.95), weight decay 0.1 on weight matrices, grad-clip 1.0, 2% warm-up then cosine to 10% |
| learning rate | peak `1e-3 * sqrt(batch / 64)`, capped at `2e-3` (so every run uses the same rule) |
| precision | fp16 autocast + loss scaling on a T4 (no fast bf16 there); the running state between layers is kept in fp32 |
| batch | no gradient accumulation - "batch size" means what one GPU step really holds |
| seeds | same seed and same data order for every run |

**Why an 8k vocabulary?** With GPT-2's 50k vocabulary the embedding table alone would be about 16M of the
20M parameters, leaving almost nothing for the layers. Reversibility works on the layers, so I wanted the
20M budget to be spent there.

## 5. What is in this repo

```
revlm/
  model.py         the GPT and the four update rules
  reversible.py    the memory-saving backward pass (the core idea, ~100 lines)
  data.py          TinyStories -> BPE -> one flat file of exactly 50M tokens
  train.py         training loop; records loss, tokens/s, peak memory
  bench.py         short speed/memory tests, max-batch search, depth scaling
  experiment.py    settings + resumable runs shared by the notebooks
  report.py        plots and the summary table (built from results/*.json)
  selftest.py      correctness checks (run at the top of every notebook)
notebooks/         01, 02, 03 (built by tools/build_notebooks.py)
tests/             pytest versions of the correctness checks
results/           written by the notebooks: one JSON per run, plots, summary.md
```

## 6. How to reproduce

1. Put this folder in a GitHub repo. Open `notebooks/01_baseline.ipynb` in Google Colab
   (`File > Open notebook > GitHub`), choose **Runtime > Change runtime type > GPU**.
2. In the first cell edit `REPO_URL` to your repo, then *Run all*. The first cell installs two small
   libraries, and the second runs the self-test (about 30 seconds) - stop if it says a check failed.
3. Run notebook 02, then notebook 03. They read the batch size and the winning variant from `results/`.
4. Every run saves its own JSON, so if Colab disconnects you can run again and finished runs are skipped.
   Set `USE_DRIVE = True` to keep data and results on Google Drive across sessions (the 50M-token file is
   rebuilt in a few minutes otherwise).
5. When done: `File > Save a copy in GitHub` for each notebook so the version with outputs is in the repo,
   and commit the `results/` folder.

`SMOKE = True` at the top of a notebook runs a tiny CPU version (about 1M parameters, 0.3M tokens) to test
the code. Its numbers mean nothing.

## 7. What I have verified so far (CPU only, no GPU available when this was written)

Run `python -m pytest tests` (7 tests) or `python -m revlm.selftest`.

| Check | Result |
|---|---|
| All four modes have the same parameter count and identical initial weights | passes |
| Rebuilding a layer's input from its output (float64) | max error about 1e-15 for all three variants |
| Memory-saving backward vs ordinary autograd, gradients of every parameter (float64) | relative difference about 1e-15 for all three variants |
| Same check under mixed precision (bf16 on CPU) | gradient cosine similarity 1.00000 |
| Chunked loss vs normal loss (see 8.3) | identical value and gradients |
| Max-batch search logic against a fake GPU with known limits | finds the limit to within 8 |
| Whole notebooks executed end to end in `SMOKE` mode | all three run without errors |
| Tiny CPU training, all four modes | all four learn (loss falls from 9.0 to about 3.7) and none diverges |

**Memory, measured without a GPU.** I counted the bytes autograd keeps for the backward pass ("saved
tensors"). This is not peak GPU memory, but it is hardware independent:

| Number of blocks (tiny test model) | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|
| baseline, bytes saved | 507,844 | 923,972 | 1,756,228 | 3,420,740 |
| reversible (midpoint), bytes saved | 110,148 | 110,148 | 110,148 | 110,148 |

For the real 20M model (ctx 512, bf16 autocast as a stand-in for fp16) the extra saved bytes per extra
token were **232.6 KiB/token for the baseline vs 68.4 KiB/token for the reversible modes**. Most of the
reversible number is the output logits (`vocab x tokens`), not the layers - see 8.3.

**What is not verified yet:** the GPU path itself - fp16 with loss scaling on a T4, the real out-of-memory
search, and real speed. The self-test at the top of each notebook re-checks the gradient agreement on the
GPU (including fp16) before any long run starts.

## 8. Design notes and honest caveats

### 8.1 "Euler" and the transcript
The session transcript is auto-generated and garbles the names ("boiler", "oiler"). I read the instructor
as asking for midpoint against an Euler-style rule. The paper's reversible Euler-style scheme is the
Hamiltonian (symplectic Euler) one, so that is what `hamiltonian` is. I added `leapfrog` because it is the
paper's third rule and costs one extra run. If your instructor meant something else, the three variants are
one flag (`mode`) apart.

### 8.2 Weight decay and dropout
The transcript says reversibility rules out weight decay and dropout. I believe only half of that is right.
Dropout does conflict, because a random mask would have to be replayed exactly when rebuilding the input, so I use none.
Weight decay only changes the weights *after* the backward pass, and the rebuild inside a step uses the same
weights as the forward pass in that step, so it does not conflict. I therefore use weight decay 0.1 in
**all** runs, so the comparison stays fair. The same transcript is auto-generated, so this may also be a
transcription slip.

### 8.3 The output logits become the bottleneck
Reversibility removes the per-layer activations. It does not touch the logits (`tokens x 8,192 x` a few
bytes), which for a 20M-parameter model are a large share of what is left. So the maximum batch will not
grow without limit. Notebook 03 therefore also tries a **chunked loss**: logits are computed a slice at a time
and recomputed in the backward pass, so the full logits never exist at once. The maths is unchanged (tested);
only memory and a little speed change. I report both.

### 8.4 A huge batch means few steps
For a fixed 50M tokens, a batch of 400 sequences (about 205k tokens) gives only about 240 optimizer steps.
The loss after the same number of *tokens* will very likely be worse than at a small batch. That would be a
property of large-batch training, not a fault of reversibility. Notebook 03 trains once with the learning
rate scaled up as `sqrt(batch)` and once with it unscaled, so the effect is visible.

### 8.5 Speed comparison is only fair at equal batch
Reversible steps at the same batch are slower (extra forward per layer). Reversible training only wins on
speed if the larger batch it allows raises GPU utilisation enough. I therefore report tokens/s at the fixed
batch (notebook 02) and at each variant's own maximum (notebook 03) separately.

### 8.6 Choices not taken from the paper
Both starting states are the embedding (`x[0] = x[1]`); the network output is the latest state (for
`hamiltonian` the `p` stream); each transformer block is one reversible step. These are reasonable choices,
not verified against the paper's code. Small-model results here are not a claim about large models.

## 9. Results

> **PENDING** - fill from `results/summary.md` after running the notebooks on Colab.

### 9.1 Batch sizes found

| model | max batch (sequences of 512 tokens) | how found |
|---|---:|---|
| baseline | PENDING | `results/maxbatch_baseline.json` |
| reversible (best variant) | PENDING | `results/maxbatch_rev_plainloss.json` |
| reversible + chunked loss | PENDING | `results/maxbatch_rev_chunkedloss.json` |

### 9.2 All runs

| run | mode | batch | steps | tokens seen | final train loss | final val loss | tokens/s (median) | peak mem (GiB) | wall time (min) | est. cost ($) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01_baseline | baseline | PENDING | | | | | | | | |
| 02_midpoint | midpoint | | | | | | | | | |
| 02_leapfrog | leapfrog | | | | | | | | | |
| 02_hamiltonian | hamiltonian | | | | | | | | | |
| 03_rev_maxbatch | (winner) | | | | | | | | | |
| 03_rev_maxbatch_lr_unscaled | (winner) | | | | | | | | | |

`tokens/s` is the median over training steps after a 20-step warm-up, including data loading and excluding
validation. `peak mem` is `torch.cuda.max_memory_allocated()` during training steps only. `est. cost` uses
the price per GPU-hour I typed into notebook 03 - it is an assumption, not a measurement.

### 9.3 Plots (written to `results/`)

- `loss_curves.png` - training loss against tokens seen, all runs
- `speed_memory.png` - tokens/s and peak memory per run
- `depth_scaling.png`, `ctx_scaling.png` - peak memory against number of blocks / sequence length, baseline vs winner

PENDING: embed the images here once they exist.

## 10. Findings

PENDING. Questions the results should answer:

1. Which reversible variant reached the lowest validation loss at the same batch, and by how much compared to the baseline? Did any diverge?
2. How much slower is a reversible step at equal batch, and does the paper's 30-50% figure hold on this GPU?
3. How much did peak memory fall at equal batch, and does it stay flat as the number of blocks grows?
4. How much larger a batch fits, and what stops it (logits, optimizer state, something else)? What does the chunked loss add?
5. What did the giant batch do to final loss, and how much of that gap was the learning-rate rule?
6. Cost of one 50M-token run with and without reversibility, at the price I assumed.

### Expectations written before the GPU runs (to compare against later)

- **[Certain]** memory saved for the backward pass stays flat as blocks are added (section 7 shows this).
- **[Likely]** reversible steps are about 25-40% slower at equal batch, because each block's forward runs twice; on the blocks alone one training step costs about 4 units of compute instead of 3 (forward + re-forward + backward, versus forward + backward).
- **[Likely]** the largest batch grows by roughly 3x, limited by the logits, and by clearly more with the chunked loss (from the 232.6 vs 68.4 KiB/token measurement above).
- **[Guessing]** the absolute batch sizes on a 16 GB T4 (my rough guess: baseline around 64-128, reversible a few hundred), and which variant wins on loss; the CPU toy runs are too small to rank them.

## 11. References

- *Reversing Large Language Models for Efficient Training and Fine-Tuning*, arXiv:2512.02056 - the midpoint, leapfrog and Hamiltonian reversible rules used here.
- ERA V5 Session 13 transcript - assignment text and the reversibility discussion.
- TinyStories dataset, `roneneldan/TinyStories` on Hugging Face.
