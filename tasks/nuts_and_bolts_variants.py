# tasks/nuts_and_bolts_variants.py
from __future__ import annotations
import os, random, math, functools, tempfile
import itertools
import pandas as pd
from typing import Dict, Any, List, Sequence
from renderer.languages.deterministic import DeterministicRenderer

from .common_variance import (
    RNG, pick_varying_dims, assign_ref_values_for_dim,
    freeze_controls, validate_trial_tiles, render_program_to_png,
    write_trial_jsonl,
)

# === Discrete value spaces ===
OUTER_TYPES: Sequence[str]       = ['c', 's', 'p', 'h']   # outer rim shape
OUTER_RING_VARIANT: Sequence[str]= ['double', 'outer_only']  # NEW: both rings vs only the large ring
NESTED_TYPES: Sequence[str]      = ['c', 's', 'p', 'h']   # center type
NESTED_SCALES: Sequence[float]   = [1.0, 2.2]             # center size
INNER_TYPES: Sequence[str]       = ['c', 's', 'p', 'h']   # ring primitive
INNER_N: Sequence[int]           = [4, 6]                 # ring count

# 2, 4, 6 dims ladder
D1_ABS = ["nested_shape_type","nested_shape_scale"]                 # 2 dims
D2_ABS = D1_ABS + ["inner_shape_type","inner_n_shapes"]             # 4 dims
D3_ABS = D2_ABS + ["outer_shape_type","outer_ring_variant"]         # 6 dims (replaces inner_phase)

VALUE_SPACE: Dict[str, Sequence[Any]] = {
    "outer_shape_type": OUTER_TYPES,
    "outer_ring_variant": OUTER_RING_VARIANT,  # NEW
    "nested_shape_type": NESTED_TYPES,
    "nested_shape_scale": NESTED_SCALES,
    "inner_shape_type": INNER_TYPES,
    "inner_n_shapes": INNER_N,
}
ALL_DIMS = list(VALUE_SPACE.keys())

def _base_features() -> Dict[str, Any]:
    return {
        "outer_shape_type": random.choice(OUTER_TYPES),
        "outer_ring_variant": random.choice(OUTER_RING_VARIANT),  # NEW
        "nested_shape_type": random.choice(NESTED_TYPES),
        "nested_shape_scale": random.choice(NESTED_SCALES),
        "inner_shape_type": random.choice(INNER_TYPES),
        "inner_n_shapes": random.choice(INNER_N),
    }

def _apothem(scale: float, n_sides: int) -> float:
    return scale * math.cos(math.pi / n_sides)

# ---- program-building helpers without side-effectful Task inits ----
@functools.lru_cache(maxsize=1)
def _nab_task_for_programs():
    tmp = os.path.join(tempfile.gettempdir(), "vlm_dummy")
    os.makedirs(tmp, exist_ok=True)
    from tasks.nuts_and_bolts import NutsAndBoltsTask
    return NutsAndBoltsTask(task_name="nab_dummy", data_dir=tmp, dataset_dir_override=tmp)

def _fit_ring_params(t, f: Dict[str, Any], dim_level: int) -> tuple[float, float]:
    canvas = t.canvas_bound
    center_scale = float(f["nested_shape_scale"])
    ring_elem = 0.40
    margin = 0.18

    if dim_level >= 3:
        n_sides_outer = t.polygon_map[f["outer_shape_type"]]
        outer_small_scale = canvas - 0.8
        outer_ap = _apothem(outer_small_scale, n_sides_outer)
        max_radius = outer_ap - 1.08*ring_elem - margin
    else:
        max_radius = canvas - 0.9

    min_radius = 1.05*center_scale + 1.08*ring_elem + margin

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

    radius = min_radius if max_radius < min_radius else (min_radius + 0.55*(max_radius - min_radius))
    return radius, ring_elem

def _build_program_for_dims(f: Dict[str, Any], dim_level: int) -> str:
    t = _nab_task_for_programs()
    canvas_bound = t.canvas_bound

    # center (always)
    center = t.generate_shape_program(f["nested_shape_type"], f["nested_shape_scale"], is_radial=False)
    parts = [center]

    # ring (dims >=2) — fixed phase (0) now; we removed the inner_phase dimension
    ring_prog = None
    if dim_level >= 2:
        radius, ring_elem_scale = _fit_ring_params(t, f, dim_level)
        base_shape = t.generate_shape_program(f["inner_shape_type"], ring_elem_scale, is_radial=True)

        n = int(f["inner_n_shapes"])
        step = (2 * math.pi) / n
        phase = 0.0  # no phase manipulation anymore

        # Keep the safe transform layout (no nested expr inside M)
        positioned = f"(T {base_shape} (M 1 {phase} {radius} 0))"
        ring_prog = f"(repeat {positioned} {n} (M 1 {step} 0 0))"

    # outer rim (dims >=3) — optionally remove one of the two rings
    outer_prog = None
    if dim_level >= 3:
        outer_scale_large = canvas_bound - 0.45
        outer_scale_small = canvas_bound - 0.8
        o1 = t.generate_shape_program(f["outer_shape_type"], outer_scale_small)  # inner ring
        o2 = t.generate_shape_program(f["outer_shape_type"], outer_scale_large)  # outer ring

        variant = f.get("outer_ring_variant", "double")
        if variant == "outer_only":
            outer_prog = o2
        else:
            outer_prog = f"(C {o1} {o2})"

    program = parts[0]
    if ring_prog is not None:
        program = f"(C {program} {ring_prog})"
    if outer_prog is not None:
        program = f"(C {outer_prog} {program})"
    return program

# ---------- exact base_task label + base-style summary ----------
from PIL import Image
from tasks.base_task import Task as _BaseTask

def _add_red_label_exact(image_path: str, label: str) -> None:
    _BaseTask._add_label_to_image(image_path, label, image_path)

def _save_summary_base_style(tile_paths: List[str], oddball_idx: int, out_path: str) -> None:
    imgs = []
    for i, p in enumerate(tile_paths, start=1):
        img = Image.open(p).convert("RGB")
        color = "red" if i == oddball_idx else "gray"
        w = 10
        bordered = Image.new("RGB", (img.width + 2*w, img.height + 2*w), color)
        bordered.paste(img, (w, w))
        imgs.append(bordered)

    if not imgs: return
    w, h = imgs[0].size
    grid = Image.new("RGB", (3*w, 2*h), "white")
    pos = [(0,0),(w,0),(2*w,0),(0,h),(w,h),(2*w,h)]
    for im, xy in zip(imgs, pos):
        grid.paste(im, xy)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    grid.save(out_path)

# ---------- render & csv ----------
def _render_tile_from_program(program_string: str, out_png: str) -> None:
    renderer = DeterministicRenderer()
    render_program_to_png(program_string, renderer, out_png, pad=5.2)
    # add exact base labels (red, 75pt)
    label = str(os.path.splitext(os.path.basename(out_png))[0].split("_")[-1])
    _add_red_label_exact(out_png, label)

def _render_tile(f: Dict[str, Any], dim_level: int, out_png: str) -> None:
    _render_tile_from_program(_build_program_for_dims(f, dim_level), out_png)

def _write_trials_csv(outdir: str, rows: list[dict]) -> None:
    import pandas as pd
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
        abstractions = D1_ABS; max_var = len(D1_ABS) - 1  # 1
    elif n_dimensions == 2:
        abstractions = D2_ABS; max_var = len(D2_ABS) - 1  # 3
    else:
        abstractions = D3_ABS; max_var = len(D3_ABS) - 1  # 5
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
        from .common_variance import choose_oddball_nonoddball_values, pick_oddball_value_excluding_refs
        oddf = dict(ref_tiles[0]["features"])
        patch = choose_oddball_nonoddball_values(ref_tiles, varying_dims, oddball_abstraction)
        for d, v in patch.items(): oddf[d] = v
        ref_vals_set = {t_["features"][oddball_abstraction] for t_ in ref_tiles}
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
        _save_summary_base_style(img_paths, oddball_idx, summary_path)

        val = validate_trial_tiles(tiles, oddball_idx, oddball_abstraction, abstractions, reference_variance)
        rows.append(dict(
            trial_idx=t_idx, summary_path=summary_path,
            oddball_idx=oddball_idx, oddball_abstraction=oddball_abstraction,
            reference_variance=reference_variance, n_dimensions=n_dimensions,
            abstraction_columns=abstractions, **val, tiles=tiles,
        ))
    write_trial_jsonl(outdir, rows)
    _write_trials_csv(outdir, rows)

def _features_to_program(f: Dict[str, Any], dim_level: int) -> str:
    # tiny wrapper so we can reuse your render builder
    return _build_program_for_dims(f, dim_level)

# --- Hydra wrappers ---
from pathlib import Path
import pandas as pd
from tasks.base_task import Task  # ensure present

class _BaseNABTask(Task):
    renderer = DeterministicRenderer()

    def __init__(self, output_dir: str, n_dimensions: int, reference_variance: int,
                 n_trials: int, seed: int = 1248, **kwargs):
        tn = kwargs.pop("task_name", None)
        output_path = Path(output_dir)
        kwargs['data_dir'] = str(output_path.parent)
        kwargs['dataset_dir_override'] = str(output_path)
        super().__init__(task_name=(tn or "nuts_and_bolts"), **kwargs)

        self.task_root_name = output_path.name
        self.output_dir = output_path
        self.n_dimensions = int(n_dimensions)
        self.reference_variance = int(reference_variance)
        self.n_trials = int(n_trials)
        self.seed = int(seed)
        self.trials_metadata_path = self.output_dir / "trials.csv"

    def generate_programs(self) -> pd.DataFrame:
        if self.n_dimensions == 1:
            abstractions = D1_ABS
        elif self.n_dimensions == 2:
            abstractions = D2_ABS
        else:
            abstractions = D3_ABS

        defaults = {k: VALUE_SPACE[k][0] for k in VALUE_SPACE}

        grids = []
        for combo in itertools.product(*[VALUE_SPACE[d] for d in abstractions]):
            f = dict(defaults)
            for d, v in zip(abstractions, combo):
                f[d] = v
            grids.append(f)

        rows = []
        for f in grids:
            prog = _features_to_program(f, self.n_dimensions)
            rows.append({"program_string": prog, **{d: f[d] for d in abstractions}})

        return pd.DataFrame(rows)

class NutsAndBoltsCenterTask(_BaseNABTask):
    def run(self):
        self.n_dimensions = 1
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "trials").mkdir(exist_ok=True)
        (self.output_dir / "summaries").mkdir(exist_ok=True)

        meta_path = self.output_dir / "metadata.csv"
        if self.overwrite_existing or (not meta_path.exists()):
            self._generate_and_render_stimuli()
        else:
            print(f"✅ Found existing metadata at {meta_path}. Skipping stimuli render…")

        need_trials = self.overwrite_existing or (not self.trials_metadata_path.exists())
        if need_trials:
            run_generate(str(self.output_dir), self.n_trials, self.reference_variance, 1, self.seed)
        else:
            print(f"✅ Found existing trials at {self.trials_metadata_path}. Skipping trial gen…")

        meta_df = pd.read_csv(meta_path)
        trials_df = pd.read_csv(self.trials_metadata_path)
        return meta_df, trials_df

class NutsAndBoltsCenterRingTask(_BaseNABTask):
    def run(self):
        self.n_dimensions = 2
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "trials").mkdir(exist_ok=True)
        (self.output_dir / "summaries").mkdir(exist_ok=True)

        meta_path = self.output_dir / "metadata.csv"
        if self.overwrite_existing or (not meta_path.exists()):
            self._generate_and_render_stimuli()
        else:
            print(f"✅ Found existing metadata at {meta_path}. Skipping stimuli render…")

        need_trials = self.overwrite_existing or (not self.trials_metadata_path.exists())
        if need_trials:
            run_generate(str(self.output_dir), self.n_trials, self.reference_variance, 2, self.seed)
        else:
            print(f"✅ Found existing trials at {self.trials_metadata_path}. Skipping trial gen…")

        meta_df = pd.read_csv(meta_path)
        trials_df = pd.read_csv(self.trials_metadata_path)
        return meta_df, trials_df

class NutsAndBoltsBothTask(_BaseNABTask):
    def run(self):
        self.n_dimensions = 3
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "trials").mkdir(exist_ok=True)
        (self.output_dir / "summaries").mkdir(exist_ok=True)

        meta_path = self.output_dir / "metadata.csv"
        if self.overwrite_existing or (not meta_path.exists()):
            self._generate_and_render_stimuli()
        else:
            print(f"✅ Found existing metadata at {meta_path}. Skipping stimuli render…")

        need_trials = self.overwrite_existing or (not self.trials_metadata_path.exists())
        if need_trials:
            run_generate(str(self.output_dir), self.n_trials, self.reference_variance, 3, self.seed)
        else:
            print(f"✅ Found existing trials at {self.trials_metadata_path}. Skipping trial gen…")

        meta_df = pd.read_csv(meta_path)
        trials_df = pd.read_csv(self.trials_metadata_path)
        return meta_df, trials_df