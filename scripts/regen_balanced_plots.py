import os
import re
import json
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter


BASE = Path("/scratch/gpfs/nb0564/vlm_reasoning")
OUTP = BASE / "plots" / "publication"
OUTP.mkdir(parents=True, exist_ok=True)
BAL_OUT = BASE / "balanced_output"
BAL_DATA = BASE / "balanced_data"

DOMAINS = [
    ("regular_polygons", r"^regular_polygons_(\d)dim_var(\d+)_dim\d+\.csv$", "None/gemini-flash"),
    ("glyphs", r"^glyphs_dim(\d)_var(\d+)_dim\d+\.csv$", "glyphs/gemini-flash"),
    ("nuts_and_bolts", r"^nab_dim(\d)_var(\d+)_dim\d+\.csv$", "nuts_and_bolts/gemini-flash"),
    ("totems", r"^totems_(\d)mod_var(\d+)_dim\d+\.csv$", "totems/gemini-flash"),
]


def read_trials_map(path: Path):
    if not path or not path.exists():
        return {}
    mapping = {}
    with open(path) as f:
        for ln in f:
            try:
                j = json.loads(ln)
                mapping[int(j["trial_idx"])] = int(j["oddball_idx"])
            except Exception:
                pass
    return mapping


def load_balanced_accuracy(mode: str) -> pd.DataFrame:
    rec = []
    for domain, pat_str, out_rel in DOMAINS:
        pat = re.compile(pat_str)
        out_dir = BAL_OUT / mode / out_rel
        if not out_dir.exists():
            continue
        for f in sorted(out_dir.glob("*.csv")):
            m = pat.search(f.name)
            if not m:
                continue
            lvl, var = int(m.group(1)), int(m.group(2))
            data_root = BAL_DATA / mode / domain
            trials = None
            for cand in [
                data_root / f"dim{lvl}" / f"var{var}" / "trials.jsonl",
                data_root / f"{lvl}mod" / f"var{var}" / "trials.jsonl",
            ]:
                if cand.exists():
                    trials = cand
                    break
            idx2odd = read_trials_map(trials)
            df = pd.read_csv(f)
            # Identify model response column
            resp_col = None
            for c in ("response", "model_response", "prediction", "pred", "model_answer", "model_pred"):
                if c in df.columns:
                    resp_col = c
                    break
            if "trial_idx" not in df.columns:
                df = df.reset_index().rename(columns={"index": "trial_idx"})
            # Ground truth oddball index
            if "oddball_idx" in df.columns:
                gt = df["oddball_idx"]
            else:
                df["__odd"] = df["trial_idx"].map(lambda t: idx2odd.get(int(t)))
                gt = df["__odd"]
            if resp_col is None:
                # Cannot compute accuracy without a model response
                continue
            # Parse numeric indices from responses and gt, handling strings like "tile_3"
            resp_str = df[resp_col].astype(str).str.strip()
            resp_num = pd.to_numeric(resp_str.str.extract(r"(\d+)")[0], errors="coerce")
            gt_num = pd.to_numeric(pd.Series(gt).astype(str).str.extract(r"(\d+)")[0], errors="coerce")
            # Heuristic alignment for 1-based vs 0-based
            if resp_num.notna().any() and gt_num.notna().any():
                resp_min, resp_max = resp_num.min(), resp_num.max()
                gt_min, gt_max = gt_num.min(), gt_num.max()
                if (
                    pd.notna(resp_min)
                    and pd.notna(resp_max)
                    and pd.notna(gt_min)
                    and pd.notna(gt_max)
                    and 1 <= resp_min <= 6
                    and 1 <= resp_max <= 6
                    and 0 <= gt_min <= 5
                    and 0 <= gt_max <= 5
                ):
                    resp_num = resp_num - 1
            corr = (resp_num.astype("Int64") == gt_num.astype("Int64")).astype("Int64").fillna(0).astype(int)
            rec.append(
                pd.DataFrame(
                    {
                        "domain": domain,
                        "dim": lvl,
                        "var": var,
                        "trial_idx": df["trial_idx"],
                        "is_correct": corr,
                    }
                )
            )
    return (
        pd.concat(rec, ignore_index=True)
        if rec
        else pd.DataFrame(columns=["domain", "dim", "var", "trial_idx", "is_correct"])
    )


def ci_boot(values, B=2500, alpha=0.05):
    a = np.asarray(values, float)
    if a.size == 0:
        return (np.nan, np.nan, np.nan)
    boots = [np.mean(np.random.choice(a, size=a.size, replace=True)) for _ in range(B)]
    return float(a.mean()), float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))


def style_axes(ax, title):
    ax.set_title(title, pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(round(y * 100))}%"))
    ax.grid(axis="y", which="major", color="#b0b0b0", alpha=0.24, linewidth=0.6)
    ax.grid(axis="y", which="minor", alpha=0.14, linewidth=0.4)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy", labelpad=5)
    ax.margins(x=0.04)


def main():
    # Accuracy vs dim (balanced dim)
    acc_dimB = load_balanced_accuracy("dim_balanced")
    print("dim_balanced rows:", len(acc_dimB))
    P1 = []
    for dim, g in sorted(acc_dimB.groupby("dim")):
        m, lo, hi = ci_boot(g["is_correct"].values)
        P1.append((dim, m, lo, hi, len(g)))
    P1 = (
        pd.DataFrame(P1, columns=["dim", "mean", "lo", "hi", "n"]).sort_values("dim")
        if P1
        else pd.DataFrame(columns=["dim", "mean", "lo", "hi", "n"])
    )
    fig, ax = plt.subplots(figsize=(3.7, 3.2))
    if not P1.empty:
        ax.plot(P1["dim"], P1["mean"], color="#1f78b4", marker="o", markersize=4.2, lw=1.7)
        ax.fill_between(P1["dim"], P1["lo"], P1["hi"], color="#1f78b4", alpha=0.15, linewidth=0)
        for _, r in P1.iterrows():
            ax.annotate(
                f"n={int(r['n'])}",
                (r["dim"], r["mean"]),
                xytext=(0, -10),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=7,
                color="#777777",
            )
    ax.set_xlabel("Dim")
    style_axes(ax, "Accuracy vs Dimension")
    fig.savefig(OUTP / "balanced_accuracy_vs_dim.pdf")
    fig.savefig(OUTP / "balanced_accuracy_vs_dim.png", dpi=200)
    plt.close(fig)
    print("Saved:", OUTP / "balanced_accuracy_vs_dim.pdf")
    print("P1:\n", P1)

    # Accuracy vs var (balanced var)
    acc_varB = load_balanced_accuracy("var_balanced")
    print("var_balanced rows:", len(acc_varB))
    P2 = []
    for var, g in sorted(acc_varB.groupby("var")):
        m, lo, hi = ci_boot(g["is_correct"].values)
        P2.append((var, m, lo, hi, len(g)))
    P2 = (
        pd.DataFrame(P2, columns=["var", "mean", "lo", "hi", "n"]).sort_values("var")
        if P2
        else pd.DataFrame(columns=["var", "mean", "lo", "hi", "n"])
    )
    fig, ax = plt.subplots(figsize=(3.7, 3.2))
    if not P2.empty:
        ax.plot(P2["var"], P2["mean"], color="#6F4C9B", marker="o", markersize=4.2, lw=1.7)
        ax.fill_between(P2["var"], P2["lo"], P2["hi"], color="#6F4C9B", alpha=0.15, linewidth=0)
        for _, r in P2.iterrows():
            ax.annotate(
                f"n={int(r['n'])}",
                (r["var"], r["mean"]),
                xytext=(0, -10),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=7,
                color="#777777",
            )
    ax.set_xlabel("Var")
    style_axes(ax, "Accuracy vs Variance")
    fig.savefig(OUTP / "balanced_accuracy_vs_var.pdf")
    fig.savefig(OUTP / "balanced_accuracy_vs_var.png", dpi=200)
    plt.close(fig)
    print("Saved:", OUTP / "balanced_accuracy_vs_var.pdf")
    print("P2:\n", P2)


if __name__ == "__main__":
    main()


