from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Any, Sequence, Set
import random, os, json

RNG = random.Random()

# ----------------- variance helpers -----------------
def choose_ref_partition() -> List[int]:
    # Always produce 5 refs as 3-of-A and 2-of-B
    return [3, 2]

def _two_distinct_values(value_space: Sequence[Any], base_val: Any) -> tuple[Any, Any]:
    # returns (A, B) two distinct values; prefers choosing A != base
    uniq = list(dict.fromkeys(value_space))
    if len(uniq) == 1:
        return uniq[0], uniq[0]
    # choose A as a non-base if available, else any
    non_base = [v for v in uniq if v != base_val]
    if not non_base:
        # all values equal to base (degenerate)
        return base_val, base_val
    A = RNG.choice(non_base)
    # choose B distinct from A; prefer base if base != A (so we get alt+base)
    candidates = [v for v in uniq if v != A]
    if base_val in candidates:
        B = base_val
    else:
        B = RNG.choice(candidates)
    return A, B

def assign_ref_values_for_dim(value_space: Sequence[Any], base_val: Any) -> List[Any]:
    """
    Build exactly 5 assignments for refs with at least 2 distinct values overall.
    Pattern: 3 of value A and 2 of value B, then shuffle.
    A is chosen (when possible) to differ from base_val, B is then chosen != A
    (prefer B = base_val when possible).
    """
    a, b = _two_distinct_values(value_space, base_val)
    part = choose_ref_partition()  # [3, 2]
    vals: List[Any] = [a] * part[0] + [b] * part[1]
    RNG.shuffle(vals)
    return vals  # always length 5

def pick_varying_dims(abstraction_columns: List[str],
                      oddball_abstraction: str,
                      reference_variance: int) -> List[str]:
    pool = [d for d in abstraction_columns if d != oddball_abstraction]
    v = min(max(reference_variance, 0), len(pool))
    return RNG.sample(pool, v)

def freeze_controls(base_features: Dict[str, Any],
                    all_dims: List[str],
                    abstraction_columns: List[str]) -> Dict[str, Any]:
    return {d: base_features[d] for d in all_dims if d not in abstraction_columns}

def validate_trial_tiles(tiles: List[Dict[str, Any]],
                         oddball_idx: int,
                         oddball_abstraction: str,
                         abstraction_columns: List[str],
                         reference_variance: int) -> Dict[str, Any]:
    ref_tiles = [t for t in tiles if t["index"] != oddball_idx]
    all6_by_dim, refs_by_dim = {}, {}
    for dim in abstraction_columns:
        all6_by_dim[dim] = Counter([t["features"][dim] for t in tiles])
        refs_by_dim[dim] = Counter([t["features"][dim] for t in ref_tiles])

    ref_varying = [d for d, cnt in refs_by_dim.items()
                   if d != oddball_abstraction and len(cnt) > 1]
    ref_constant = [d for d, cnt in refs_by_dim.items()
                    if d != oddball_abstraction and len(cnt) == 1]

    nonodd_singletons = []
    for d, cnt in all6_by_dim.items():
        if d == oddball_abstraction:
            continue
        if any(v == 1 for v in cnt.values()):
            nonodd_singletons.append(d)

    return dict(
        ref_varying_dims_actual=ref_varying,
        ref_constant_dims_actual=ref_constant,
        ref_variance_count_actual=len(ref_varying),
        ref_singleton_columns=[],
        all_nonoddball_singleton_columns=nonodd_singletons,
        all6_value_counts=all6_by_dim,
        ref_value_counts=refs_by_dim,
    )

# ---------- DSL render → PNG ----------
from renderer.core import parse_program
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _is_colored(stroke):
    return isinstance(stroke, tuple) and len(stroke) == 2 and isinstance(stroke[1], tuple)

def render_program_to_png(program_string: str, renderer, out_path: str,
                          pad: float = 5.5) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig = plt.figure(figsize=(4, 4), dpi=200)
    ax = plt.gca()
    ax.set_xlim(-pad, pad); ax.set_ylim(-pad, pad)
    ax.set_aspect("equal"); ax.axis("off")
    if program_string:
        ast = parse_program(program_string)
        strokes = renderer.evaluate(ast)
        for s in strokes:
            if _is_colored(s):
                arr, col = s
                ax.plot(arr[:,0], arr[:,1], linewidth=1.4, color=col)
            elif isinstance(s, np.ndarray):
                ax.plot(s[:,0], s[:,1], linewidth=1.4, color="black")
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

# ---------- per-trial 6-up summary (annotated) ----------
def save_trial_summary(tile_paths: List[str], oddball_idx: int, out_path: str) -> None:
    import matplotlib.image as mpimg
    import matplotlib.patches as patches
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig = plt.figure(figsize=(6, 4), dpi=200)

    for i, p in enumerate(tile_paths, start=1):
        ax = fig.add_subplot(2, 3, i)
        ax.axis("off")
        try:
            img = mpimg.imread(p)
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, "missing", ha="center", va="center")
        # index label
        ax.text(0.05, 0.9, str(i), transform=ax.transAxes, fontsize=10,
                bbox=dict(facecolor='white', edgecolor='black', pad=1.5, lw=0.8))
        # red box around oddball
        if i == oddball_idx:
            rect = patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                     fill=False, edgecolor='red', linewidth=2.0)
            ax.add_patch(rect)

    fig.subplots_adjust(wspace=0.02, hspace=0.02)
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

# ---------- oddball helpers ----------
def _counts(values: List[Any]) -> Dict[Any, int]:
    return dict(Counter(values))

def choose_oddball_nonoddball_values(ref_tiles: List[Dict[str, Any]],
                                     varying_dims: List[str],
                                     oddball_abstraction: str) -> Dict[str, Any]:
    choice = {}
    for d in varying_dims:
        if d == oddball_abstraction:
            continue
        vals = [t["features"][d] for t in ref_tiles]
        counts = _counts(vals)
        # pick the rarest among refs so that, once oddball is added, no singleton remains
        v_sorted = sorted(counts.items(), key=lambda kv: (kv[1], str(kv[0])))
        choice[d] = v_sorted[0][0]
    return choice

def pick_oddball_value_excluding_refs(value_space: Sequence[Any],
                                      ref_values: Set[Any],
                                      current_value: Any) -> Any:
    # prefer a value not seen in refs; else any value != current_value
    pool = [v for v in value_space if v not in ref_values]
    if pool:
        return RNG.choice(pool)
    alts = [v for v in value_space if v != current_value]
    return RNG.choice(alts) if alts else current_value

# ---------- jsonl utility ----------
def write_trial_jsonl(outdir: str, rows: List[Dict[str, Any]]) -> None:
    meta_path = os.path.join(outdir, "trials.jsonl")
    with open(meta_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

@dataclass
class TrialTile:
    index: int
    image_path: str
    features: Dict[str, Any]
