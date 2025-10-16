from __future__ import annotations
import os, random
from typing import Dict, Any, List, Sequence
from tasks.totems import TotemsTask
from renderer.languages.colored import ColoredRenderer

from .common_variance import (
    RNG, pick_varying_dims, assign_ref_values_for_dim,
    freeze_controls, validate_trial_tiles,
    render_program_to_png, write_trial_jsonl, save_trial_summary,
    choose_oddball_nonoddball_values, pick_oddball_value_excluding_refs
)

SHAPES: Sequence[str] = ['circle', 'triangle', 'square']
COLORS: Sequence[str] = ['red', 'green', 'blue']

D1_ABS  = ["module1_shape", "module1_color"]
D2_ABS  = D1_ABS + ["module2_shape","module2_color"]
D3_ABS  = D2_ABS + ["module3_shape","module3_color"]

VALUE_SPACE: Dict[str, Sequence[Any]] = {
    "module1_shape": SHAPES, "module1_color": COLORS,
    "module2_shape": SHAPES, "module2_color": COLORS,
    "module3_shape": SHAPES, "module3_color": COLORS,
}
ALL_DIMS = list(VALUE_SPACE.keys())

def _base_features() -> Dict[str, Any]:
    return {
        "module1_shape": random.choice(SHAPES), "module1_color": random.choice(COLORS),
        "module2_shape": random.choice(SHAPES), "module2_color": random.choice(COLORS),
        "module3_shape": random.choice(SHAPES), "module3_color": random.choice(COLORS),
    }

def _record_to_program(f: Dict[str, Any], n_dimensions: int) -> str:
    t = TotemsTask()
    if n_dimensions == 1:
        modules = ((f["module1_shape"], f["module1_color"]),)
    elif n_dimensions == 2:
        modules = ((f["module1_shape"], f["module1_color"]),
                   (f["module2_shape"], f["module2_color"]))
    else:
        modules = ((f["module1_shape"], f["module1_color"]),
                   (f["module2_shape"], f["module2_color"]),
                   (f["module3_shape"], f["module3_color"]))
    rec = t._create_program_record(modules)
    return rec["program_string"]

def _render_tile(rec: Dict[str, Any], n_dimensions: int, out_png: str) -> None:
    renderer = ColoredRenderer()
    render_program_to_png(_record_to_program(rec, n_dimensions), renderer, out_png)

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

    abstractions = D1_ABS if n_dimensions == 1 else D2_ABS if n_dimensions == 2 else D3_ABS
    reference_variance = min(max(reference_variance, 0), len(abstractions)-1)

    rows = []
    for t in range(n_trials):
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

        # oddball with singleton elimination and exclusion from ref values
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
            out_png = os.path.join(outdir, "trials", f"trial={t}_{pos}.png")
            _render_tile(f, n_dimensions, out_png)
            tiles.append(dict(index=pos, image_path=out_png, features=f.copy()))
            img_paths.append(out_png)

        summary_path = os.path.join(outdir, "summaries", f"summary=trial{t}.png")
        save_trial_summary(img_paths, oddball_idx, summary_path)

        val = validate_trial_tiles(tiles, oddball_idx, oddball_abstraction, abstractions, reference_variance)
        rows.append(dict(
            trial_idx=t, summary_path=summary_path,
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

class _BaseTotemsTask(Task):
    renderer = ColoredRenderer()

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
        super().__init__(task_name=(tn or "totems"), **kwargs)

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

class TotemsOneModuleTask(_BaseTotemsTask):
    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "trials").mkdir(exist_ok=True)
        (self.output_dir / "summaries").mkdir(exist_ok=True)
        run_generate(str(self.output_dir), self.n_trials, self.reference_variance, 1, self.seed)

class TotemsTwoModuleTask(_BaseTotemsTask):
    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "trials").mkdir(exist_ok=True)
        (self.output_dir / "summaries").mkdir(exist_ok=True)
        run_generate(str(self.output_dir), self.n_trials, self.reference_variance, 2, self.seed)

class TotemsThreeModuleTask(_BaseTotemsTask):
    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "trials").mkdir(exist_ok=True)
        (self.output_dir / "summaries").mkdir(exist_ok=True)
        run_generate(str(self.output_dir), self.n_trials, self.reference_variance, 3, self.seed)
