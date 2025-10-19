# tasks/base_task.py
from __future__ import annotations

import json
import random
import traceback
from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# renderer core (confirmed signatures)
from renderer.core import (
    parse_program,           # (program_string: str) -> AstNode
    render_strokes_to_image, # (strokes: list, canvas_dim:int=512, coord_bound:float=5.0, line_width:float=3.0) -> np.ndarray
    export_image,            # (image_array: np.ndarray, export_path: str) -> None
)


class Task(ABC):
    """
    Abstract base task.
    Subclasses must set:
      - renderer: instance exposing .evaluate(ast) -> strokes (list-like)
      - generate_programs(): DataFrame with 'program_string' and abstraction columns
    """

    @property
    @abstractmethod
    def renderer(self):
        pass

    def __init__(
        self,
        task_name: str,
        data_dir: str = "data",
        n_trials: int = 100,
        n_dimensions: int = 3,
        reference_variance: int = 1,
        stroke_width: float = 3.0,
        overwrite_existing: bool = False,
        dup_factor: int = 1,
        canvas_dim: int = 512,
        coord_bound: float = 5.0,
        exclude_abstraction_keywords: str | list[str] | None = None,
        # NEW: let callers force the exact dataset directory (variants use this)
        dataset_dir_override: str | None = None,
        **kwargs,
    ):
        self.task_name = task_name
        self.data_dir = Path(data_dir)
        self.n_trials = int(n_trials)
        self.n_dimensions = int(n_dimensions)
        self.reference_variance = int(reference_variance)
        self.stroke_width = float(stroke_width)
        self.overwrite_existing = bool(overwrite_existing)
        self.dup_factor = int(dup_factor)
        self.canvas_dim = int(canvas_dim)
        self.coord_bound = float(coord_bound)
        self._dataset_dir_override = Path(dataset_dir_override) if dataset_dir_override else None

        # normalize exclude keywords; supports str (comma/space separated) or list[str]
        if exclude_abstraction_keywords is None:
            self.exclude_keywords: list[str] = []
        elif isinstance(exclude_abstraction_keywords, str):
            parts: list[str] = []
            for chunk in exclude_abstraction_keywords.replace(",", " ").split():
                c = chunk.strip().lower()
                if c:
                    parts.append(c)
            self.exclude_keywords = parts
        else:
            self.exclude_keywords = [
                str(s).strip().lower()
                for s in exclude_abstraction_keywords
                if str(s).strip()
            ]

        # paths + dirs + params.json
        self._setup_paths()
        self._create_directories()
        # legacy field used by some model codepaths
        self.task_root_name = self.dataset_dir.name

    # ---------------- paths ----------------

    def _setup_paths(self):
        if self._dataset_dir_override is not None:
            # exact path (variants pass their output_dir here)
            self.dataset_dir = self._dataset_dir_override
        else:
            # Nested layout: data/<task_name>/dimX/varY
            self.dataset_dir = (
                self.data_dir
                / self.task_name
                / f"dim{self.n_dimensions}"
                / f"var{self.reference_variance}"
            )
        self.images_dir = self.dataset_dir / "images"
        self.trials_dir = self.dataset_dir / "trials"
        self.summaries_dir = self.dataset_dir / "summaries"

        self.metadata_path = self.dataset_dir / "metadata.csv"
        self.trials_metadata_path = self.dataset_dir / "trials.csv"
        self.params_path = self.dataset_dir / "params.json"

    def _create_directories(self):
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.trials_dir.mkdir(parents=True, exist_ok=True)
        self.summaries_dir.mkdir(parents=True, exist_ok=True)
        self.params_path.parent.mkdir(parents=True, exist_ok=True)
        params = {
            "task_name": self.task_name,
            "n_dimensions": self.n_dimensions,
            "reference_variance": self.reference_variance,
            "n_trials": self.n_trials,
            "stroke_width": self.stroke_width,
            "dup_factor": self.dup_factor,
            "canvas_dim": self.canvas_dim,
            "coord_bound": self.coord_bound,
            "exclude_keywords": self.exclude_keywords,
            "data_dir": str(self.data_dir),
            "dataset_dir": str(self.dataset_dir),
        }
        with open(self.params_path, "w") as f:
            json.dump(params, f, indent=2)

    # --------------- abstract ---------------

    @abstractmethod
    def generate_programs(self) -> pd.DataFrame:
        """Return a DataFrame with at least 'program_string' and abstraction columns."""
        raise NotImplementedError

    # --------------- pipeline ---------------

    def run(self):
        """
        Safe base pipeline:

        - If BOTH metadata.csv and trials.csv exist and overwrite is false: load & return.
        - Otherwise, ensure metadata.csv exists (render stimuli if needed).
        - If trials.csv exists and overwrite is false: load trials and return with metadata.
        - Else: build trials via the base pipeline and generate summaries.
          (Variants that manage trials themselves should NOT call this method; they
           should call _generate_and_render_stimuli() and then their custom generator.)
        """
        if self.overwrite_existing:
            self._clear_existing_outputs()

        meta_exists = self.metadata_path.exists()
        trials_exists = self.trials_metadata_path.exists()

        if (not self.overwrite_existing) and meta_exists and trials_exists:
            print(f"✅ Found existing metadata and trials at {self.dataset_dir}. Loading…")
            return pd.read_csv(self.metadata_path), pd.read_csv(self.trials_metadata_path)

        # ensure metadata
        if not meta_exists:
            print("🎨 Generating stimuli and metadata…")
            stimuli_df = self._generate_and_render_stimuli()
        else:
            print(f"📄 Using existing metadata at {self.metadata_path}")
            stimuli_df = pd.read_csv(self.metadata_path)

        # if trials already exist and we are not overwriting, just load them
        if trials_exists and (not self.overwrite_existing):
            print(f"✅ Found existing trials at {self.trials_metadata_path}. Loading…")
            trials_df = pd.read_csv(self.trials_metadata_path)
            return stimuli_df, trials_df

        # otherwise, proceed with base trial generation
        print("🧪 Building trials via base pipeline…")
        trials_df = self._generate_all_oddball_trials(stimuli_df)
        self._generate_trial_summaries(trials_df)
        return stimuli_df, trials_df

    def _clear_existing_outputs(self):
        try:
            for p in (self.metadata_path, self.trials_metadata_path):
                if p.exists():
                    p.unlink()
            for d in (self.images_dir, self.trials_dir, self.summaries_dir):
                if d.exists():
                    for f in d.glob("*"):
                        try:
                            f.unlink()
                        except Exception:
                            pass
        except Exception as e:
            print(f"⚠️ Could not fully clear outputs: {e}")

    # ---------- step 1: render ----------

    def _render_one(self, program_string: str) -> np.ndarray:
        ast = parse_program(program_string)          # AstNode
        strokes = self.renderer.evaluate(ast)        # list-like strokes
        img_arr = render_strokes_to_image(           # to bitmap
            strokes,
            canvas_dim=self.canvas_dim,
            coord_bound=self.coord_bound,
            line_width=self.stroke_width,
        )
        return img_arr

    def _generate_and_render_stimuli(self) -> pd.DataFrame:
        df = self.generate_programs().copy()
        assert "program_string" in df.columns, "generate_programs() must include 'program_string'"

        filepaths = []
        for i, row in tqdm(df.iterrows(), total=len(df), desc="🎨 Rendering stimuli", unit="stimulus"):
            out_path = self.images_dir / f"{i}.png"
            try:
                img_arr = self._render_one(str(row["program_string"]))
                export_image(img_arr, str(out_path))
                filepaths.append(str(out_path))
            except Exception as e:
                print(f"⚠️ Rendering failed for idx={i}: {e}")
                filepaths.append(None)

        df["render_filepath"] = filepaths
        df = df.dropna(subset=["render_filepath"]).reset_index(drop=True)
        df.to_csv(self.metadata_path, index=False)
        print(f"✅ Saved stimuli metadata to: {self.metadata_path}")
        return df

    # ---------- helpers ----------

    @staticmethod
    def _to_py(x: Any) -> Any:
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if pd.isna(x):
            return None
        return x

    def _row_features(self, row: pd.Series, cols: Iterable[str]) -> dict:
        return {c: self._to_py(row[c]) for c in cols if c in row.index}

    # ---------- step 2: trials ----------

    def get_abstraction_columns(self, metadata_df: pd.DataFrame) -> list[str]:
        """
        Default: all non-admin columns truncated by n_dimensions.
        Then apply keyword-based include/exclude filters (substring match).
        """
        admin = {"program_string", "render_filepath", "abstraction"}
        cols = [c for c in metadata_df.columns if c not in admin]
        cols = cols[: self.n_dimensions]

        # Exclude if the column name contains any of the keywords
        if getattr(self, "exclude_keywords", None):
            lowered = [c.lower() for c in cols]
            keep = []
            for c, lc in zip(cols, lowered):
                if any(k in lc for k in self.exclude_keywords):
                    continue
                keep.append(c)
            if keep:
                cols = keep

        return cols

    def _generate_all_oddball_trials(self, metadata_df: pd.DataFrame) -> pd.DataFrame:
        abstraction_cols = self.get_abstraction_columns(metadata_df)
        print(f"🧭 Using abstraction columns: {abstraction_cols}")

        records = []
        per_abs = self.n_trials
        total = per_abs * len(abstraction_cols)
        pbar = tqdm(total=total, desc="🧪 Building trials", unit="trial")

        for abstraction in abstraction_cols:
            done = 0
            attempts = 0
            max_attempts = int(os.environ.get("MAX_ATTEMPTS_PER_ABS", "50000"))
            base_counts = metadata_df[abstraction].value_counts().to_dict()
            while done < per_abs:
                rec = self._create_single_trial(metadata_df, abstraction, abstraction_cols, trial_idx=len(records))
                attempts += 1
                if rec is not None:
                    records.append(rec)
                    done += 1
                    pbar.update(1)
                elif attempts % 5000 == 0:
                    print(f"\n   ↳ still searching for {abstraction} (attempts={attempts}). value_counts={base_counts}")
                if attempts >= max_attempts:
                    print(f"⚠️ Giving up on {abstraction} after {attempts} attempts. Collected {done}/{per_abs} trials for this abstraction.")
                    break
        pbar.close()

        trials_df = pd.DataFrame.from_records(records)
        if not trials_df.empty:
            trials_df.to_csv(self.trials_metadata_path, index=False)
            print(f"✅ Saved trials index to: {self.trials_metadata_path}")
        else:
            print("⚠️ No trials generated.")
        return trials_df

    def _create_single_trial(self, df: pd.DataFrame, abstraction: str, all_abstractions: list[str], trial_idx: int) -> dict | None:
        try:
            target_value = random.choice(df[abstraction].unique().tolist())
            refs = self._sample_reference_stimuli(df, abstraction, target_value, all_abstractions)
            if refs is None:
                return None
            odd = self._find_oddball_stimulus(df, refs, abstraction, target_value)
            if odd is None or odd.empty:
                return None

            # Final guard: ensure no-singleton across *six* for non-oddball dims
            if not self._six_no_singleton_ok(refs, odd.iloc[0], abstraction, all_abstractions):
                return None

            return self._process_and_save_trial_assets(refs, odd, abstraction, trial_idx, all_abstractions)
        except Exception as e:
            print(f"❌ Error generating trial for {abstraction}: {e}")
            print(traceback.format_exc())
            return None

    def _sample_reference_stimuli(
        self,
        df: pd.DataFrame,
        abstraction: str,
        value: Any,
        used_abstractions: list[str],
        max_attempts: int = 20_000
    ) -> pd.DataFrame | None:
        """
        Return 5 references with:
        - target abstraction fixed to `value`
        - exactly `reference_variance` other dims varying (or fewer if infeasible)
        - for each varying dim among the 5 refs: no singletons (each value count ≥2)
        - all other non-oddball dims constant
        """
        base = df[df[abstraction] == value]
        if base.empty:
            return None

        if self.dup_factor > 1:
            base = pd.concat([base] * self.dup_factor, ignore_index=True)

        non_target = [c for c in used_abstractions if c != abstraction]

        # Which dims are even capable of varying
        feasibles = []
        for c in non_target:
            vc = base[c].value_counts()
            if vc.nunique() >= 1 and vc.index.size >= 2 and vc.max() >= 2:
                feasibles.append(c)

        import itertools, random as _random
        K = min(self.reference_variance, len(feasibles))
        vary_sets = list(itertools.combinations(feasibles, K)) if K > 0 else [tuple()]

        for _ in range(max_attempts):
            varying = set(_random.choice(vary_sets))
            constants = [c for c in non_target if c not in varying]

            # choose a concrete value for every constant dim
            const_vals = {c: _random.choice(base[c].unique().tolist()) for c in constants}

            # filter pool to rows that actually satisfy constant values
            pool = base.copy()
            for c, v in const_vals.items():
                pool = pool[pool[c] == v]
                if pool.empty:
                    break
            if pool.empty:
                continue

            # get candidate values per varying dim (each value must have >=2 availability)
            choices = {}
            for c in varying:
                avail = pool[c].value_counts()
                popular = [v for v, cnt in avail.items() if cnt >= 2]
                if len(popular) < 2:
                    choices = None
                    break
                vals = (_random.sample(popular, 3) if len(popular) >= 3 and _random.random() < 0.3
                        else _random.sample(popular, 2))
                choices[c] = vals
            if choices is None:
                continue

            # construct a multi-set of 5 assignments for each varying column (no singletons)
            col_bags = {}
            for c, vals in choices.items():
                if len(vals) == 3:
                    col = [vals[0], vals[0], vals[1], vals[1], vals[2]]  # provisional 2/2/1
                else:
                    v1, v2 = vals
                    col = [v1, v1, v1, v2, v2]  # 3/2
                _random.shuffle(col)
                col_bags[c] = col

            # realize 5 actual rows consistent with all column assignments simultaneously
            taken_idx = set()
            chosen_rows = []
            for i in range(5):
                sub = pool
                for c in varying:
                    sub = sub[sub[c] == col_bags[c][i]]
                    if sub.empty:
                        break
                if sub.empty:
                    chosen_rows = []
                    break
                sub = sub[~sub.index.isin(taken_idx)]
                if sub.empty:
                    # allow reuse as last resort
                    sub = pool
                    for c in varying:
                        sub = sub[sub[c] == col_bags[c][i]]
                pick = sub.sample(n=1).iloc[0]
                chosen_rows.append(pick)
                taken_idx.add(pick.name)

            if not chosen_rows:
                continue

            sample = pd.DataFrame(chosen_rows)

            # verify constraints across the 5 references
            ok = True
            if any(sample[c].nunique() != 1 for c in constants):
                ok = False
            for c in varying:
                vc = sample[c].value_counts()
                if sample[c].nunique() < 2 or vc.min() < 2:
                    ok = False
                    break

            if ok:
                return sample

        return None


    def _find_oddball_stimulus(
        self,
        df: pd.DataFrame,
        reference: pd.DataFrame,
        abstraction: str,
        target_value: Any
    ) -> pd.DataFrame | None:
        """
        Pick an oddball differing on `abstraction`, matching allowed sets on others,
        and preserving no-singleton across the 6 images.
        """
        non_target_cols = [
            c for c in reference.columns
            if c not in {"program_string", "render_filepath"} and c != abstraction
        ]
        allowed = {c: set(reference[c].unique()) for c in non_target_cols}

        mask = (df[abstraction] != target_value)
        for c, vals in allowed.items():
            mask &= df[c].isin(vals)
        cands = df[mask]
        if cands.empty:
            return None

        cands = cands.sample(n=min(len(cands), 256), replace=False)
        for _, row in cands.iterrows():
            ok = True
            for c in non_target_cols:
                combined = reference[c].tolist() + [row[c]]
                vc = pd.Series(combined).value_counts()
                if (vc == 1).any():
                    ok = False
                    break
            if ok:
                if "program_string" in reference.columns and row.get("program_string") in set(reference["program_string"]):
                    continue
                return row.to_frame().T

        return None

    def _six_no_singleton_ok(self, refs: pd.DataFrame, odd_row: pd.Series, oddball_abs: str, all_abs: list[str]) -> bool:
        for c in all_abs:
            if c == oddball_abs:
                continue
            combined = refs[c].tolist() + [odd_row[c]]
            vc = pd.Series(combined).value_counts()
            if vc.min() < 2:
                return False
        return True

    def _process_and_save_trial_assets(
        self,
        refs: pd.DataFrame,
        oddball: pd.DataFrame,
        oddball_abstraction: str,
        trial_idx: int,
        abstraction_cols: list[str],
    ) -> dict:
        refs = refs.sample(frac=1.0, random_state=random.randint(0, 1_000_000)).reset_index(drop=True)
        odd_row = oddball.iloc[0]
        ref_paths = refs["render_filepath"].tolist()
        odd_path = odd_row["render_filepath"]

        # slot the oddball
        odd_pos = random.randint(1, 6)
        ordered_paths = []
        ordered_feats = []
        rptr = 0
        for slot in range(1, 7):
            if slot == odd_pos:
                ordered_paths.append(odd_path)
                ordered_feats.append(self._row_features(odd_row, abstraction_cols))
            else:
                ordered_paths.append(ref_paths[rptr])
                ordered_feats.append(self._row_features(refs.iloc[rptr], abstraction_cols))
                rptr += 1

        # save six tiles with clean red numerals (original style)
        for i, src in enumerate(ordered_paths, 1):
            dst = self.trials_dir / f"trial={trial_idx}_{i}.png"
            self._add_label_to_image(src, str(i), dst)

        # counts across 6 and across refs
        def _counts(vals):
            s = pd.Series(vals)
            vc = s.value_counts(dropna=False)
            return {str(k) if k is not None else None: int(v) for k, v in vc.items()}

        all6_value_counts = {c: _counts([f.get(c) for f in ordered_feats]) for c in abstraction_cols}
        ref_value_counts  = {c: _counts([f.get(c) for i,f in enumerate(ordered_feats,1) if i != odd_pos]) for c in abstraction_cols}

        ref_varying = [c for c, vc in ref_value_counts.items() if len([k for k,v in vc.items() if v>0]) > 1]
        ref_singletons = [c for c, vc in ref_value_counts.items() if any(v == 1 for v in vc.values())]
        nonodd_singletons = [c for c in abstraction_cols if c != oddball_abstraction and any(v == 1 for v in all6_value_counts[c].values())]

        # sidecar JSON (rich)
        tiles = []
        for i, feats in enumerate(ordered_feats, 1):
            tiles.append({
                "index": i,
                "image_path": str(self.trials_dir / f"trial={trial_idx}_{i}.png"),
                "features": feats,
            })

        rec = {
            "trial_idx": trial_idx,
            "oddball_idx": odd_pos,
            "oddball_abstraction": oddball_abstraction,
            "reference_variance": self.reference_variance,
            "n_dimensions": self.n_dimensions,
            "abstraction_columns": abstraction_cols,
            "ref_varying_dims_actual": ref_varying,
            "ref_constant_dims_actual": [c for c in abstraction_cols if c not in ref_varying],
            "ref_variance_count_actual": len(ref_varying),
            "ref_singleton_columns": ref_singletons,
            "all_nonoddball_singleton_columns": nonodd_singletons,
            "all6_value_counts": all6_value_counts,
            "ref_value_counts": ref_value_counts,
            "tiles": tiles,
        }
        with open(self.trials_dir / f"trial={trial_idx}_meta.json", "w") as f:
            json.dump(rec, f, indent=2)

        return rec

    # ---------- step 3: summaries ----------

    def _generate_trial_summaries(self, trials_df: pd.DataFrame):
        if trials_df.empty:
            print("⚠️ No trials generated, skipping visualizations")
            return
        for rec in tqdm(trials_df.to_dict("records"), desc="🖼️ Generating example trial visualizations"):
            self._save_trial_grid(rec["trial_idx"], rec["oddball_idx"])

    def _save_trial_grid(self, trial_idx: int, oddball_idx: int):
        tiles = []
        for i in range(1, 7):
            img_path = self.trials_dir / f"trial={trial_idx}_{i}.png"
            img = Image.open(img_path).convert("RGB")
            color = "red" if i == oddball_idx else "gray"   # ORIGINAL styling
            tiles.append(self._add_border_to_image(img, color, width=10))

        w, h = tiles[0].width, tiles[0].height
        grid = Image.new("RGB", (3 * w, 2 * h), "white")
        positions = [(0, 0), (w, 0), (2*w, 0), (0, h), (w, h), (2*w, h)]
        for tile, pos in zip(tiles, positions):
            grid.paste(tile, pos)

        grid.save(self.summaries_dir / f"trial={trial_idx}_grid.png")

    @staticmethod
    def _add_border_to_image(image: Image.Image, color: str, width: int = 10) -> Image.Image:
        out = Image.new("RGB", (image.width + 2 * width, image.height + 2 * width), color)
        out.paste(image, (width, width))
        return out

    @staticmethod
    def _add_label_to_image(image_path: Path | str, text: str, output_path: Path | str):
        from PIL import Image, ImageDraw, ImageFont
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        W, H = img.size
        # ~2x smaller than the previous 0.22 scaling
        font_px = max(24, int(0.05 * min(W, H)))
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_px)
        except Exception:
            font = ImageFont.load_default()
        margin = int(0.04 * min(W, H))
        draw.text((margin, margin), text, fill="red", font=font)
        img.save(output_path)