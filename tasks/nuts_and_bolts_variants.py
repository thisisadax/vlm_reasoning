from __future__ import annotations
import os, random, math
from typing import Dict, Any, List, Sequence
from tasks.nuts_and_bolts import NutsAndBoltsTask
from renderer.languages.deterministic import DeterministicRenderer

from .common_variance import (
    RNG, pick_varying_dims, assign_ref_values_for_dim,
    freeze_controls, validate_trial_tiles,
    render_program_to_png, write_trial_jsonl, save_trial_summary,
    choose_oddball_nonoddball_values, pick_oddball_value_excluding_refs
)

# === Safer, clearer ranges (no overlaps / on-canvas) ===
OUTER_TYPES: Sequence[str]     = ['c', 's', 'p', 'h']
NESTED_TYPES: Sequence[str]    = ['c', 's', 'p', 'h']    # central type
NESTED_SCALES: Sequence[float] = [1.0, 2.2]             # small / large (reduced spread)
INNER_TYPES: Sequence[str]     = ['c', 's', 'p', 'h']    # ring primitive
INNER_N: Sequence[int]         = [4, 6]                 # clearer counts

D1_ABS = ["nested_shape_type","nested_shape_scale"]
D2_ABS = D1_ABS + ["inner_shape_type","inner_n_shapes"]
D3_ABS = D2_ABS + ["outer_shape_type"]

VALUE_SPACE: Dict[str, Sequence[Any]] = {
    "outer_shape_type": OUTER_TYPES,
    "nested_shape_type": NESTED_TYPES,
    "nested_shape_scale": NESTED_SCALES,
    "inner_shape_type": INNER_TYPES,
    "inner_n_shapes": INNER_N,
}
ALL_DIMS = list(VALUE_SPACE.keys())

def _base_features() -> Dict[str, Any]:
    return {
        "outer_shape_type": random.choice(OUTER_TYPES),
        "nested_shape_type": random.choice(NESTED_TYPES),
        "nested_shape_scale": random.choice(NESTED_SCALES),
        "inner_shape_type": random.choice(INNER_TYPES),
        "inner_n_shapes": random.choice(INNER_N),
    }

def _apothem(scale: float, n_sides: int) -> float:
    return scale * math.cos(math.pi / n_sides)

# --- robust containment for ring (no clip with center or rim) ---
def _fit_ring_params(t: NutsAndBoltsTask, f: Dict[str, Any], dim_level: int) -> tuple[float, float]:
    """
    Returns (radius, ring_elem_scale) that:
      - clears central nested shape by margin
      - if dim3, remains strictly inside the small outer rim apothem by margin
      - stays within canvas
    """
    canvas = t.canvas_bound                    # ~5.0
    center_scale = float(f["nested_shape_scale"])
    ring_elem = 0.40                           # start smaller
    margin = 0.18                              # larger safety

    # compute max radius feasible bound
    if dim_level >= 3:
        n_sides_outer = t.polygon_map[f["outer_shape_type"]]
        outer_small_scale = canvas - 0.8       # slightly smaller rim
        outer_ap = _apothem(outer_small_scale, n_sides_outer)
        max_radius = outer_ap - 1.08*ring_elem - margin
    else:
        max_radius = canvas - 0.9

    # min radius must clear center + ring thickness
    min_radius = 1.05*center_scale + 1.08*ring_elem + margin

    # If infeasible, shrink ring element stepwise until feasible (down to 0.22)
    while (min_radius > max_radius) and (ring_elem > 0.22):
        ring_elem -= 0.03
        if dim_level >= 3:
            n_sides_outer = t.polygon_map[f["outer_shape_type"]]
            outer_small_scale = canvas - 0.8
            outer_ap = _apothem(outer_small_scale, n_sides_outer)
            max_radius = outer_ap - 1.08*ring_elem - margin
        else:
            max_radius = canvas - 0.9
        min_radius = 1.05*center_scale + 1.08*ring_elem + margin

    # choose a middle-ish radius, leaned away from boundaries
    if max_radius < min_radius:
        radius = min_radius
    else:
        radius = min_radius + 0.55 * (max_radius - min_radius)
    return radius, ring_elem

def _build_program_for_dims(f: Dict[str, Any], dim_level: int) -> str:
    t = NutsAndBoltsTask()
    canvas_bound = t.canvas_bound

    # center (nested only for dim1)
    center = t.generate_shape_program(f["nested_shape_type"], f["nested_shape_scale"], is_radial=False)
    parts = [center]

    ring_prog = None
    if dim_level >= 2:
        radius, ring_elem_scale = _fit_ring_params(t, f, dim_level)
        base_shape = t.generate_shape_program(f["inner_shape_type"], ring_elem_scale, is_radial=True)
        positioned = f"(T {base_shape} (M 1 0 {radius} 0))"
        angle = (2 * math.pi) / int(f["inner_n_shapes"])
        ring_prog = f"(repeat {positioned} {int(f['inner_n_shapes'])} (M 1 {angle} 0 0))"

    outer_prog = None
    if dim_level >= 3:
        outer_scale_large = canvas_bound - 0.45    # slightly pulled in
        outer_scale_small = canvas_bound - 0.8
        o1 = t.generate_shape_program(f["outer_shape_type"], outer_scale_small)
        o2 = t.generate_shape_program(f["outer_shape_type"], outer_scale_large)
        outer_prog = f"(C {o1} {o2})"

    program = parts[0]
    if ring_prog is not None:
        program = f"(C {program} {ring_prog})"
    if outer_prog is not None:
        program = f"(C {outer_prog} {program})"
    return program

def _render_tile(f: Dict[str, Any], dim_level: int, out_png: str) -> None:
    renderer = DeterministicRenderer()
    render_program_to_png(_build_program_for_dims(f, dim_level), renderer, out_png, pad=5.2)

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

def run_generate(outdir: str, n_trials: int, reference_variance: int,
                 n_dimensions: int, seed: int) -> None:
    RNG.seed(seed); random.seed(seed)
    os.makedirs(os.path.join(outdir, "trials"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "summaries"), exist_ok=True)

    if n_dimensions == 1:
        abstractions = D1_ABS; max_var = 1
    elif n_dimensions == 2:
        abstractions = D2_ABS; max_var = 3
    else:
        abstractions = D3_ABS; max_var = 4
    reference_variance = min(max(reference_variance, 0), max_var)

    rows = []
    for t_idx in range(n_trials):
        base = _base_features()
        frozen = freeze_controls(base, ALL_DIMS, abstractions)
        oddball_abstraction = random.choice(abstractions)
        varying_dims = pick_varying_dims(abstractions, oddball_abstraction, reference_variance)

        ref_schedules: Dict[str, List[Any]] = {}
        for d in varying_dims:
            ref_schedules[d] = assign_ref_values_for_dim(VALUE_SPACE[d], base[d])

        # 5 refs
        ref_tiles: List[Dict[str, Any]] = []
        for _ in range(5):
            f = dict(base); f.update(frozen)
            for d in abstractions:
                f[d] = ref_schedules[d].pop() if d in varying_dims else base[d]
            ref_tiles.append(dict(features=f.copy()))

        # oddball with singleton elimination + exclude ref values
        oddf = dict(ref_tiles[0]["features"])
        patch = choose_oddball_nonoddball_values(ref_tiles, varying_dims, oddball_abstraction)
        for d, v in patch.items(): oddf[d] = v
        ref_vals_set = {t["features"][oddball_abstraction] for t in ref_tiles}
        oddf[oddball_abstraction] = pick_oddball_value_excluding_refs(
            VALUE_SPACE[oddball_abstraction], ref_vals_set, oddf[oddball_abstraction]
        )

        oddball_idx = RNG.randint(1, 6)
        tiles: List[Dict[str, Any]] = []
        img_paths: List[str] = []
        ref_iter = iter(ref_tiles)
        for pos in range(1, 7):
            f = oddf if pos == oddball_idx else next(ref_iter)["features"]
            out_png = os.path.join(outdir, "trials", f"trial={t_idx}_{pos}.png")
            _render_tile(f, n_dimensions, out_png)
            tiles.append(dict(index=pos, image_path=out_png, features=f.copy()))
            img_paths.append(out_png)

        summary_path = os.path.join(outdir, "summaries", f"summary=trial{t_idx}.png")
        save_trial_summary(img_paths, oddball_idx, summary_path)

        val = validate_trial_tiles(tiles, oddball_idx, oddball_abstraction, abstractions, reference_variance)
        rows.append(dict(
            trial_idx=t_idx, summary_path=summary_path,
            oddball_idx=oddball_idx, oddball_abstraction=oddball_abstraction,
            reference_variance=reference_variance, n_dimensions=n_dimensions,
            abstraction_columns=abstractions, **val, tiles=tiles,
        ))
    write_trial_jsonl(outdir, rows)
    _write_trials_csv(outdir, rows)

# --- Hydra-compatible Task wrappers ---
from pathlib import Path
import pandas as pd
from tasks.base_task import Task  # ensure present

class _BaseNABTask(Task):
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
        tn = kwargs.pop("task_name", None)

        # Set data_dir to output_dir.parent so model looks for images in the right place
        output_path = Path(output_dir)
        kwargs['data_dir'] = str(output_path.parent)
        super().__init__(task_name=(tn or "nuts_and_bolts"), **kwargs)

        # Override task_root_name to point to the correct directory
        self.task_root_name = output_path.name

        self.output_dir = output_path
        self.n_dimensions = int(n_dimensions)
        self.reference_variance = int(reference_variance)
        self.n_trials = int(n_trials)
        self.seed = int(seed)
        # point Hydra/inference at CSV
        self.trials_metadata_path = self.output_dir / "trials.csv"

    def generate_programs(self) -> pd.DataFrame:
        return pd.DataFrame([])

class NutsAndBoltsCenterTask(_BaseNABTask):
    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "trials").mkdir(exist_ok=True)
        (self.output_dir / "summaries").mkdir(exist_ok=True)
        run_generate(str(self.output_dir), self.n_trials, self.reference_variance, 1, self.seed)

class NutsAndBoltsCenterRingTask(_BaseNABTask):
    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "trials").mkdir(exist_ok=True)
        (self.output_dir / "summaries").mkdir(exist_ok=True)
        run_generate(str(self.output_dir), self.n_trials, self.reference_variance, 2, self.seed)

class NutsAndBoltsBothTask(_BaseNABTask):
    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "trials").mkdir(exist_ok=True)
        (self.output_dir / "summaries").mkdir(exist_ok=True)
        run_generate(str(self.output_dir), self.n_trials, self.reference_variance, 3, self.seed)
