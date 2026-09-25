"""Turn results/*.json into plots + a markdown summary table (no numbers are typed by hand)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# validated categorical order from the dataviz palette: blue, orange, aqua, yellow
COLORS = {"baseline": "#2a78d6", "midpoint": "#eb6834", "leapfrog": "#1baf7a", "hamiltonian": "#eda100"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def load_results(results_dir="results"):
    out = []
    for p in sorted(Path(results_dir).glob("*.json")):
        try:
            r = json.loads(p.read_text())
        except Exception:
            continue
        if isinstance(r, dict) and "train_losses" in r:      # skip benchmark/search files
            out.append(r)
    return out


def _mode_of(r):
    return r["model"]["mode"]


def _style(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED)


def _smooth(x, k=25):
    x = np.asarray(x, dtype=float)
    if len(x) < k:
        return x
    c = np.cumsum(np.insert(x, 0, 0))
    out = np.empty_like(x)
    for i in range(len(x)):
        lo = max(0, i - k + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def plot_loss_curves(results, path, title="Training loss vs tokens seen"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=140)
    _style(ax)
    for r in results:
        m = _mode_of(r)
        tps = r["tokens_per_step"]
        y = _smooth(r["train_losses"])
        x = (np.arange(len(y)) + 1) * tps / 1e6
        ls = "-" if m == "baseline" or r["model"].get("rev_impl") == "memory_saving" else ":"
        ax.plot(x, y, color=COLORS[m], lw=2, ls=ls,
                label=f"{r['run_name']} (B={r['batch_size']})")
    ax.set_xlabel("tokens seen (millions)", color=MUTED)
    ax.set_ylabel("train loss (25-step average)", color=MUTED)
    lo = min(min(_smooth(r["train_losses"])[50:] if len(r["train_losses"]) > 60 else r["train_losses"]) for r in results)
    ax.set_ylim(bottom=max(0, lo - 0.1), top=lo + 1.6)
    ax.set_title(title, color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_bars(results, path):
    """Two small panels: speed and peak memory per run (one axis each - never dual-axis)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rs = [r for r in results if r.get("tokens_per_sec_median")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=140)
    names = [f"{r['run_name']}\nB={r['batch_size']}" for r in rs]
    cols = [COLORS[_mode_of(r)] for r in rs]
    for ax, key, lab in ((axes[0], "tokens_per_sec_median", "tokens / second"),
                         (axes[1], "peak_mem_allocated_mib", "peak GPU memory (GiB)")):
        _style(ax)
        vals = [(r[key] or 0) / (1024 if "mem" in key else 1) for r in rs]
        bars = ax.bar(range(len(rs)), vals, color=cols, width=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.1f}" if "mem" in key else f"{v:,.0f}",
                    ha="center", va="bottom", fontsize=7, color=INK)
        ax.set_xticks(range(len(rs)))
        ax.set_xticklabels(names, fontsize=6.5, color=MUTED)
        ax.set_title(lab, color=INK, loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def summary_table(results, price_per_hour=None):
    rows = ["| run | mode | batch | steps | tokens seen | final train loss | final val loss | tokens/s (median) | peak mem (GiB) | wall time (min) |" + (" est. cost ($) |" if price_per_hour else ""),
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|" + ("---:|" if price_per_hour else "")]
    for r in results:
        pm = r.get("peak_mem_allocated_mib")
        cost = f" {r['wall_time_s']/3600*price_per_hour:.2f} |" if price_per_hour else ""
        flag = " (diverged)" if r.get("diverged") else ""
        rows.append(f"| {r['run_name']}{flag} | {_mode_of(r)} | {r['batch_size']} | {r['steps_done']} | "
                    f"{r['tokens_seen']/1e6:.1f}M | {r['final_train_loss']:.4f} | {r['final_val_loss']:.4f} | "
                    f"{r['tokens_per_sec_median']:,.0f} | {pm/1024:.2f} | {r['wall_time_s']/60:.1f} |".replace("| nan", "| n/a")
                    if pm is not None else
                    f"| {r['run_name']}{flag} | {_mode_of(r)} | {r['batch_size']} | {r['steps_done']} | "
                    f"{r['tokens_seen']/1e6:.1f}M | {r['final_train_loss']:.4f} | {r['final_val_loss']:.4f} | "
                    f"{r['tokens_per_sec_median']:,.0f} | n/a | {r['wall_time_s']/60:.1f} |")
        if price_per_hour:
            rows[-1] += cost
    return "\n".join(rows)


def write_summary(results_dir="results", price_per_hour=None):
    d = Path(results_dir)
    res = load_results(d)
    if not res:
        print("no results yet")
        return
    plot_loss_curves(res, d / "loss_curves.png")
    plot_bars(res, d / "speed_memory.png")
    (d / "summary.md").write_text(summary_table(res, price_per_hour) + "\n")
    print(summary_table(res, price_per_hour))


def plot_scaling(rows, key, path, xlabel):
    """Peak memory (GiB) vs `key` (n_layer or ctx) for each variant in `rows` (from bench.scaling_table)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = sorted({r["name"] for r in rows}, key=lambda n: (n != "baseline", n))
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=140)
    _style(ax)
    for n in names:
        pts = [(r[key], r["peak_mib"] / 1024) for r in rows if r["name"] == n and r["ok"] and r["peak_mib"]]
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", lw=2, color=COLORS.get(n, "#4a3aa7"), label=n)
        oom = [r[key] for r in rows if r["name"] == n and not r["ok"]]
        for x in oom:
            ax.annotate("OOM", (x, ax.get_ylim()[1] * 0.95), color=COLORS.get(n, MUTED), fontsize=7, ha="center")
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel("peak GPU memory (GiB)", color=MUTED)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
