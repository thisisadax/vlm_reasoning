from __future__ import annotations
import os, random, functools, tempfile
import itertools
import pandas as pd
from typing import Dict, Any, List, Sequence
from renderer.languages.probabilistic import ProbabilisticRenderer

# keep only the constraint helpers from common_variance
from .common_variance import (
    RNG, pick_varying_dims, assign_ref_values_for_dim,
    freeze_controls, validate_trial_tiles, render_program_to_png,
    write_trial_jsonl,
)

# === value spaces ===
EMB_TYPES: Sequence[str]  = ['h_line', 's_curve', 'two_dots']
EMB_COLORS: Sequence[str] = ['red', 'green', 'blue']

D1_ABS = ["emb1_type", "emb1_color"]
D2_ABS = D1_ABS + ["emb2_type","emb2_color"]
D3_ABS = D2_ABS + ["emb3_type","emb3_color"]

VALUE_SPACE: Dict[str, Sequence[Any]] = {
    "emb1_type": EMB_TYPES, "emb1_color": EMB_COLORS,
    "emb2_type": EMB_TYPES, "emb2_color": EMB_COLORS,
    "emb3_type": EMB_TYPES, "emb3_color": EMB_COLORS,
}
ALL_DIMS = list(VALUE_SPACE.keys())

def _base_features() -> Dict[str, Any]:
    return {
        "emb1_type": random.choice(EMB_TYPES), "emb1_color": random.choice(EMB_COLORS),
        "emb2_type": random.choice(EMB_TYPES), "emb2_color": random.choice(EMB_COLORS),
        "emb3_type": random.choice(EMB_TYPES), "emb3_color": random.choice(EMB_COLORS),
    }

# ---------- PURE program builder (no side-effect Task init paths) ----------
@functools.lru_cache(maxsize=1)
def _glyphs_task_for_programs():
    # use a dummy dir to avoid polluting data/ with nested dim/var
    tmp = os.path.join(tempfile.gettempdir(), "vlm_dummy")
    os.makedirs(tmp, exist_ok=True)
    from tasks.glyphs import GlyphsTask
    return GlyphsTask(task_name="glyphs_dummy", data_dir=tmp, dataset_dir_override=tmp)

def _record_to_program(f: Dict[str, Any], dim_level: int) -> str:
    t = _glyphs_task_for_programs()
    if dim_level == 1:
        top, mid, bot = f["emb1_type"], 'none', 'none'
        top_c, mid_c, bot_c = f["emb1_color"], 'red', 'red'
    elif dim_level == 2:
        top, mid, bot = f["emb1_type"], f["emb2_type"], 'none'
        top_c, mid_c, bot_c = f["emb1_color"], f["emb2_color"], 'red'
    else:
        top, mid, bot = f["emb1_type"], f["emb2_type"], f["emb3_type"]
        top_c, mid_c, bot_c = f["emb1_color"], f["emb2_color"], f["emb3_color"]
    params = ('vertical','vertical', top, mid, bot, top_c, mid_c, bot_c)
    rec = t._create_program_record(params)
    return rec["program_string"]

# ---------- exact base_task label + base-style summary ----------
from PIL import Image, ImageDraw, ImageFont
from tasks.base_task import Task as _BaseTask

def _add_red_label_exact(image_path: str, label: str) -> None:
    # use the exact helper from base_task (red, ~75pt)
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
def _render_tile(rec: Dict[str, Any], dim_level: int, out_png: str) -> None:
    renderer = ProbabilisticRenderer()
    render_program_to_png(_record_to_program(rec, dim_level), renderer, out_png)
    _add_red_label_exact(out_png, str(os.path.splitext(os.path.basename(out_png))[0].split("_")[-1]))

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

        # oddball (singleton-proof) and exclude ref values for oddball dim
        oddf = dict(ref_tiles[0]["features"])
        from .common_variance import choose_oddball_nonoddball_values, pick_oddball_value_excluding_refs
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
            out_png = os.path.join(outdir, "trials", f"trial={t}_{pos}.png")
            _render_tile(f, n_dimensions, out_png)
            tiles.append(dict(index=pos, image_path=out_png, features=f.copy()))
            img_paths.append(out_png)

        summary_path = os.path.join(outdir, "summaries", f"summary=trial{t}.png")
        _save_summary_base_style(img_paths, oddball_idx, summary_path)

        val = validate_trial_tiles(tiles, oddball_idx, oddball_abstraction, abstractions, reference_variance)
        rows.append(dict(
            trial_idx=t, summary_path=summary_path,
            oddball_idx=oddball_idx, oddball_abstraction=oddball_abstraction,
            reference_variance=reference_variance, n_dimensions=n_dimensions,
            abstraction_columns=abstractions, **val, tiles=tiles,
        ))
    write_trial_jsonl(outdir, rows)
    _write_trials_csv(outdir, rows)

# --- Hydra wrappers (ensure dataset_dir_override pins to outdir) ---
from pathlib import Path
import pandas as pd
from tasks.base_task import Task  # must exist

class _BaseGlyphsTask(Task):
    renderer = ProbabilisticRenderer()

    def __init__(self, output_dir: str, n_dimensions: int, reference_variance: int,
                 n_trials: int, seed: int = 1248, **kwargs):
        tn = kwargs.pop("task_name", None)
        output_path = Path(output_dir)
        kwargs['data_dir'] = str(output_path.parent)
        kwargs['dataset_dir_override'] = str(output_path)  # ensure no nested pollution
        super().__init__(task_name=(tn or "glyphs"), **kwargs)

        self.task_root_name = output_path.name
        self.output_dir = output_path
        self.n_dimensions = int(n_dimensions)
        self.reference_variance = int(reference_variance)
        self.n_trials = int(n_trials)
        self.seed = int(seed)
        self.trials_metadata_path = self.output_dir / "trials.csv"

    def generate_programs(self) -> pd.DataFrame:
        # Decide which abstraction columns are in play for this n_dimensions
        abstractions = D1_ABS if self.n_dimensions == 1 else (D2_ABS if self.n_dimensions == 2 else D3_ABS)

        # For any non-abstraction dims, just freeze to a default (first entry) to keep corpus manageable
        defaults = {k: VALUE_SPACE[k][0] for k in VALUE_SPACE}
        # Enumerate the full Cartesian product across the *abstraction* dims
        grids = []
        for combo in itertools.product(*[VALUE_SPACE[d] for d in abstractions]):
            f = dict(defaults)
            for d, v in zip(abstractions, combo):
                f[d] = v
            grids.append(f)

        # Build programs
        rows = []
        for f in grids:
            prog = _record_to_program(f, self.n_dimensions)
            rows.append({"program_string": prog, **{d: f[d] for d in abstractions}})

        return pd.DataFrame(rows)

class GlyphsFirstTask(_BaseGlyphsTask):
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

class GlyphsFirstSecondTask(_BaseGlyphsTask):
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

class GlyphsAllThreeTask(_BaseGlyphsTask):
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