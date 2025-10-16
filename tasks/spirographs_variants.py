from __future__ import annotations
import os, random, math
from typing import Dict, Any, List, Sequence
from tasks.spirographs import SpirographsTask
from renderer.languages.deterministic import DeterministicRenderer

from .common_variance import (
    RNG, pick_varying_dims, assign_ref_values_for_dim,
    freeze_controls, validate_trial_tiles,
    render_program_to_png, write_trial_jsonl, save_trial_summary,
    choose_oddball_nonoddball_values, pick_oddball_value_excluding_refs
)

# === Safer, clearer ranges (no off-canvas) ===
CENTRAL_PRIMS: Sequence[str]    = ['c', 's', 't', 'p', 'h']
CENTRAL_SCALES: Sequence[float] = [1.2, 2.2]     # small / large (reduced spread)
RADIAL_PRIMS: Sequence[str]     = ['c', 's', 't', 'p', 'h']
RADIAL_SCALES: Sequence[float]  = [0.55, 1.00]   # small / large (reduced)
RADIUS_LABELS: Sequence[str]    = ['small', 'large']  # we compute actual radius from label
N_REPEATS: Sequence[int]        = [4]            # frozen for clarity

CENTRAL_ABS = ["central_primitive", "central_scale"]               # dim1
RADIAL_ABS  = ["radial_primitive", "radial_scale", "radius"]       # dim2 (radius is size label)
BOTH_ABS    = CENTRAL_ABS + RADIAL_ABS                             # dim3

VALUE_SPACE: Dict[str, Sequence[Any]] = {
    "central_primitive": CENTRAL_PRIMS,
    "central_scale": CENTRAL_SCALES,
    "radial_primitive": RADIAL_PRIMS,
    "radial_scale": RADIAL_SCALES,
    "radius": RADIUS_LABELS,   # NOTE: label, not numeric
    "n_repeats": N_REPEATS,    # frozen
}
ALL_DIMS = list(VALUE_SPACE.keys())

def _base_features() -> Dict[str, Any]:
    return {
        "central_primitive": random.choice(CENTRAL_PRIMS),
        "central_scale": random.choice(CENTRAL_SCALES),
        "radial_primitive": random.choice(RADIAL_PRIMS),
        "radial_scale": random.choice(RADIAL_SCALES),
        "radius": random.choice(RADIUS_LABELS),  # label
        "n_repeats": N_REPEATS[0],
    }

def _radius_numeric(c_scale: float, r_scale: float, label: str, canvas_bound: float = 5.0) -> float:
    """
    Compute a canvas-safe numeric radius from the (small|large) label and the
    current central/radial scales. Ensures:
      radius > c_scale + k*r_scale + margin   (clears center)
      radius + k*r_scale < canvas_bound - margin  (stays on-canvas)
    """
    k = 1.05
    margin = 0.28
    inner = c_scale + k * r_scale + margin        # min feasible
    outer = (canvas_bound - margin) - k * r_scale # max feasible
    if outer < inner:  # rare degenerate; clamp
        r = inner
    else:
        t = 0.35 if label == 'small' else 0.75    # place well inside feasible band
        r = inner + t * (outer - inner)
    return float(r)

def _record_to_program(rec: Dict[str, Any]) -> str:
    mode = rec["_mode"]; nd = rec["_nd"]
    t = SpirographsTask(dimension_mode=mode, n_dimensions=nd)
    params: List[Any] = []
    if mode in ("both", "central"):
        params.append(rec["central_primitive"])
        if nd >= 2:
            params.append(rec["central_scale"])
    if (mode == "both" and nd >= 3) or (mode == "radial"):
        params.append(rec["radial_primitive"])
        if (mode == "both" and nd >= 4) or (mode == "radial" and nd >= 2):
            params.append(rec["radial_scale"])
        if (mode == "both" and nd >= 5) or (mode == "radial" and nd >= 3):
            # map label → numeric
            rnum = _radius_numeric(rec["central_scale"], rec["radial_scale"], rec["radius"])
            params.append(rnum)
        if (mode == "both" and nd >= 6) or (mode == "radial" and nd >= 4):
            params.append(rec["n_repeats"])
    built = t._create_program_record(tuple(params), nd)
    return built["program_string"]

def _render_tile(rec: Dict[str, Any], out_png: str) -> None:
    renderer = DeterministicRenderer()
    render_program_to_png(_record_to_program(rec), renderer, out_png, pad=5.2)

def _write_trials_csv(outdir: str, rows: list[dict]) -> None:
    import pandas as pd, os
    flat = []
    for r in rows:
        flat.append({
            "trial_idx": r["trial_idx"],
            "summary_path": r.get("summary_path", ""),
            "oddball_idx": r["oddball_idx"],
            "oddball_abstraction": r["oddball_abstraction"],
            "reference_variance": r["reference_variance"],
            "n_dimensions": r["n_dimensions"],
            "abstraction_columns": ",".join(r.get("abstraction_columns", [])),
        })
    pd.DataFrame(flat).sort_values("trial_idx").to_csv(os.path.join(outdir, "trials.csv"), index=False)

def _generate_trial(outdir: str, reference_variance: int,
                    dimension_mode: str, n_dimensions: int,
                    seed: int, trial_idx: int) -> Dict[str, Any]:
    base = _base_features()

    if dimension_mode == "central":
        abstractions, nd = CENTRAL_ABS, 2
    elif dimension_mode == "radial":
        abstractions, nd = RADIAL_ABS, 3
    else:
        abstractions, nd = BOTH_ABS, 5

    frozen = freeze_controls(base, ALL_DIMS, abstractions)
    oddball_abstraction = random.choice(abstractions)
    reference_variance = min(max(reference_variance, 0), len(abstractions)-1)
    varying_dims = pick_varying_dims(abstractions, oddball_abstraction, reference_variance)

    ref_schedules: Dict[str, List[Any]] = {}
    for d in varying_dims:
        ref_schedules[d] = assign_ref_values_for_dim(VALUE_SPACE[d], base[d])

    # 5 references
    ref_tiles: List[Dict[str, Any]] = []
    for _ in range(5):
        f = dict(base); f.update(frozen)
        for d in abstractions:
            f[d] = ref_schedules[d].pop() if d in varying_dims else base[d]
        ref_tiles.append(dict(features=f.copy()))

    # oddball with singleton elimination for non-oddball varying dims
    oddf = dict(ref_tiles[0]["features"])
    patch = choose_oddball_nonoddball_values(ref_tiles, varying_dims, oddball_abstraction)
    for d, v in patch.items(): oddf[d] = v

    # ensure oddball value is NOT among ref values for its abstraction (if possible)
    ref_vals_set = {t["features"][oddball_abstraction] for t in ref_tiles}
    oddf[oddball_abstraction] = pick_oddball_value_excluding_refs(
        VALUE_SPACE[oddball_abstraction], ref_vals_set, oddf[oddball_abstraction]
    )

    # randomized placement
    oddball_idx = RNG.randint(1, 6)
    tiles: List[Dict[str, Any]] = []
    img_paths: List[str] = []
    ref_iter = iter(ref_tiles)
    for pos in range(1, 7):
        f = oddf if pos == oddball_idx else next(ref_iter)["features"]
        rec = dict(f, _mode=dimension_mode, _nd=nd)
        out_png = os.path.join(outdir, "trials", f"trial={trial_idx}_{pos}.png")
        _render_tile(rec, out_png)
        tiles.append(dict(index=pos, image_path=out_png, features=f.copy()))
        img_paths.append(out_png)

    summary_path = os.path.join(outdir, "summaries", f"summary=trial{trial_idx}.png")
    save_trial_summary(img_paths, oddball_idx, summary_path)

    val = validate_trial_tiles(tiles, oddball_idx, oddball_abstraction, abstractions, reference_variance)
    return dict(
        trial_idx=trial_idx, summary_path=summary_path,
        oddball_idx=oddball_idx, oddball_abstraction=oddball_abstraction,
        reference_variance=reference_variance, n_dimensions=nd,
        abstraction_columns=abstractions, **val, tiles=tiles,
    )

def run_generate(outdir: str, n_trials: int, reference_variance: int,
                 dimension_mode: str, n_dimensions: int, seed: int) -> None:
    RNG.seed(seed); random.seed(seed)
    os.makedirs(os.path.join(outdir, "trials"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "summaries"), exist_ok=True)
    rows = []
    for t in range(n_trials):
        rows.append(_generate_trial(outdir, reference_variance, dimension_mode, n_dimensions, seed, t))
    write_trial_jsonl(outdir, rows)
    _write_trials_csv(outdir, rows)

# --- Hydra-compatible Task wrappers ---
from pathlib import Path
import pandas as pd
from tasks.base_task import Task  # ensure present

class _BaseSpiroTask(Task):
    renderer = DeterministicRenderer()

    def __init__(
        self,
        output_dir: str,
        n_dimensions: int,
        reference_variance: int,
        n_trials: int,
        seed: int = 1248,
        **kwargs,
    ):
        # Avoid duplicate task_name: pop if Hydra provided it
        tn = kwargs.pop("task_name", None)

        # Set data_dir to output_dir.parent so model looks for images in the right place
        output_path = Path(output_dir)
        kwargs['data_dir'] = str(output_path.parent)
        super().__init__(task_name=(tn or "spirographs"), **kwargs)

        # Override task_root_name to point to the correct directory
        self.task_root_name = output_path.name

        self.output_dir = output_path
        self.n_dimensions = int(n_dimensions)
        self.reference_variance = int(reference_variance)
        self.n_trials = int(n_trials)
        self.seed = int(seed)
        # point Hydra/inference at CSV
        self.trials_metadata_path = self.output_dir / "trials.csv"

    # Satisfy abstract API (not used by our custom run)
    def generate_programs(self) -> pd.DataFrame:
        return pd.DataFrame([])

class SpirographsCentralTask(_BaseSpiroTask):
    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "trials").mkdir(exist_ok=True)
        (self.output_dir / "summaries").mkdir(exist_ok=True)
        run_generate(
            outdir=str(self.output_dir),
            n_trials=self.n_trials,
            reference_variance=self.reference_variance,
            dimension_mode="central",
            n_dimensions=self.n_dimensions,  # 2
            seed=self.seed,
        )

class SpirographsRadialTask(_BaseSpiroTask):
    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "trials").mkdir(exist_ok=True)
        (self.output_dir / "summaries").mkdir(exist_ok=True)
        run_generate(
            outdir=str(self.output_dir),
            n_trials=self.n_trials,
            reference_variance=self.reference_variance,
            dimension_mode="radial",
            n_dimensions=self.n_dimensions,  # 3
            seed=self.seed,
        )

class SpirographsBothTask(_BaseSpiroTask):
    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "trials").mkdir(exist_ok=True)
        (self.output_dir / "summaries").mkdir(exist_ok=True)
        run_generate(
            outdir=str(self.output_dir),
            n_trials=self.n_trials,
            reference_variance=self.reference_variance,
            dimension_mode="both",
            n_dimensions=self.n_dimensions,  # 5
            seed=self.seed,
        )
