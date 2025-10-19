# tasks/regular_polygons_variants.py
"""Regular polygons with proper oddball control."""
from __future__ import annotations
import os, random, math, itertools, json
import pandas as pd
from typing import Dict, Any, List, Sequence
from pathlib import Path
from renderer.languages.polygonal import PolygonalRenderer
from tasks.base_task import Task
from PIL import Image

from tasks.common_variance import (
    RNG, pick_varying_dims, assign_ref_values_for_dim,
    freeze_controls, validate_trial_tiles, render_program_to_png,
    write_trial_jsonl,
)

# DIM 1: Basic shape properties (2 dimensions)
N_SIDES = [3, 4, 5, 6]  # Triangle, square, pentagon, hexagon
EDGE_TYPES = ['straight', 'convex', 'concave']  # Line, outward arc, inward arc

# DIM 2: Add transformations (4 dimensions total)
TRANSFORM_TYPES = ['radial', 'line', 'spiral', 'scale_origin', 'scale_outward']  # No 'none' to avoid phantom oddballs
N_COPIES = [3, 4, 5, 6]  # Number of copies in transformation

# DIM 3: Add compositions (6 dimensions total)  
COMPOSITION_TYPES = ['none', 'nested', 'layered', 'rotated']
COMP_SCALES = [0.4, 0.5, 0.6, 0.7]

# Dimension definitions - PROPERLY STRUCTURED
D1_ABS = ["n_sides", "edge_type"]  # 2 dimensions
D2_ABS = D1_ABS + ["transform_type", "n_copies"]  # 4 dimensions
D3_ABS = D2_ABS + ["composition_type", "comp_scale"]  # 6 dimensions

VALUE_SPACE = {
    "n_sides": N_SIDES,
    "edge_type": EDGE_TYPES,
    "transform_type": TRANSFORM_TYPES,
    "n_copies": N_COPIES,
    "composition_type": COMPOSITION_TYPES,
    "comp_scale": COMP_SCALES,
}

def _generate_closed_polygon(n_sides: int, scale: float, edge_type: str = 'straight') -> str:
    """Generate a closed polygon with specified edge type."""
    angle_per_side = 2 * math.pi / n_sides
    
    # Calculate vertices
    vertices = []
    for i in range(n_sides):
        angle = i * angle_per_side - math.pi / 2  # Start from top
        x = scale * math.cos(angle)
        y = scale * math.sin(angle)
        vertices.append((x, y))
    
    # Build edges
    strokes = []
    for i in range(n_sides):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n_sides]
        
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx*dx + dy*dy)
        angle = math.atan2(dy, dx)
        
        if edge_type == 'straight':
            edge = f"(T l (M {length:.4f} {angle:.4f} {x1:.4f} {y1:.4f}))"
        elif edge_type == 'convex':
            # Outward bulge
            arc = f"(a 0.25)"
            edge = f"(T {arc} (M {length:.4f} {angle:.4f} {x1:.4f} {y1:.4f}))"
        elif edge_type == 'concave':
            # Inward curve  
            arc = f"(a -0.25)"
            edge = f"(T {arc} (M {length:.4f} {angle:.4f} {x1:.4f} {y1:.4f}))"
        else:
            edge = f"(T l (M {length:.4f} {angle:.4f} {x1:.4f} {y1:.4f}))"
        
        strokes.append(edge)
    
    # Compose all strokes
    result = strokes[0]
    for stroke in strokes[1:]:
        result = f"(C {result} {stroke})"
    
    return result

def _apply_transform(shape: str, transform_type: str, n_copies: int) -> str:
    """Apply transformation with specified type and number of copies."""
    if transform_type == 'radial':
        # Radial arrangement
        angle_step = 2 * math.pi / n_copies
        radius = 1.8
        parts = []
        for i in range(n_copies):
            theta = i * angle_step
            x = radius * math.cos(theta)
            y = radius * math.sin(theta)
            positioned = f"(T {shape} (M 1 0 {x:.3f} {y:.3f}))"
            parts.append(positioned)
        result = parts[0]
        for p in parts[1:]:
            result = f"(C {result} {p})"
        return result
        
    elif transform_type == 'line':
        # Linear arrangement with adaptive spacing
        spacing = 2.2 / math.sqrt(n_copies / 4)
        parts = []
        for i in range(n_copies):
            offset = i * spacing - (n_copies - 1) * spacing / 2
            translated = f"(T {shape} (M 1 0 {offset:.3f} 0))"
            parts.append(translated)
        result = parts[0]
        for p in parts[1:]:
            result = f"(C {result} {p})"
        return result
        
    elif transform_type == 'spiral':
        # Spiral using repeat operator
        rotation_per_step = 2 * math.pi / n_copies + 0.2
        scale_per_step = 0.9
        return f"(repeat {shape} {n_copies} 0 0 {scale_per_step:.3f} {rotation_per_step:.5f})"
    
    elif transform_type == 'scale_origin':
        # Concentric scaling - all centered at origin
        parts = []
        for i in range(n_copies):
            scale_factor = 1.0 + i * 0.4
            scaled = f"(T {shape} (M {scale_factor:.2f} 0 0 0))"
            parts.append(scaled)
        result = parts[0]
        for p in parts[1:]:
            result = f"(C {result} {p})"
        return result
    
    elif transform_type == 'scale_outward':
        # Exponential scaling with shared anchor point (bottom-left quadrant nesting)
        # Each shape is scale_factor times larger, anchored at same origin
        scale_factor = 1.4  # Each shape 1.4x larger than previous
        parts = []
        for i in range(n_copies):
            current_scale = scale_factor ** i
            # Anchor at origin - smaller shapes nest inside larger ones
            scaled = f"(T {shape} (M {current_scale:.3f} 0 0 0))"
            parts.append(scaled)
        result = parts[0]
        for p in parts[1:]:
            result = f"(C {result} {p})"
        return result
    
    return shape

def _apply_composition(shape: str, comp_type: str, comp_scale: float) -> str:
    """Apply composition pattern."""
    if comp_type == 'none':
        return shape
    elif comp_type == 'nested':
        inner = f"(T {shape} (M {comp_scale:.2f} 0 0 0))"
        inner2 = f"(T {shape} (M {comp_scale*comp_scale:.2f} 0 0 0))"
        return f"(C (C {shape} {inner}) {inner2})"
    elif comp_type == 'layered':
        layer2 = f"(T {shape} (M {comp_scale:.2f} 0.7854 0 0))"
        return f"(C {shape} {layer2})"
    elif comp_type == 'rotated':
        rot1 = f"(T {shape} (M 1 1.0472 0 0))"
        rot2 = f"(T {shape} (M 1 2.0944 0 0))"
        return f"(C (C {shape} {rot1}) {rot2})"
    return shape

def _calculate_canvas_scale(features: Dict[str, Any], n_dimensions: int) -> float:
    """Calculate scaling to keep everything within canvas."""
    complexity = 1.0
    
    # Base shape
    n_sides = features.get("n_sides", 4)
    if n_sides == 3:
        complexity *= 1.1
    elif n_sides >= 6:
        complexity *= 0.95
    
    # Edge type
    edge_type = features.get("edge_type", "straight")
    if edge_type == "convex":
        complexity *= 1.15
    elif edge_type == "concave":
        complexity *= 0.9
    
    # Transform (D2+)
    if n_dimensions >= 2:
        transform = features.get("transform_type", "radial")
        n_copies = features.get("n_copies", 4)
        
        if transform == "radial":
            complexity *= 2.2
        elif transform == "line":
            complexity *= 1.8 + n_copies * 0.3
        elif transform == "spiral":
            complexity *= 1.8
        elif transform == "scale_origin":
            complexity *= 1.0 + n_copies * 0.3
        elif transform == "scale_outward":
            # Exponential growth needs more aggressive scaling
            complexity *= 1.4 ** (n_copies - 1)
    
    # Composition (D3)
    if n_dimensions >= 3:
        comp = features.get("composition_type", "none")
        if comp != "none":
            complexity *= 1.2
    
    # Target 60% canvas for safety
    target_extent = 3.0
    return min(0.75, target_extent / (complexity + 0.5))

def _build_program(features: Dict[str, Any], n_dimensions: int) -> str:
    """Build program from features."""
    # Base shape
    n_sides = features.get("n_sides", 4)
    edge_type = features.get("edge_type", "straight")
    base = _generate_closed_polygon(n_sides, 1.0, edge_type)
    
    # Apply transform (D2+)
    if n_dimensions >= 2:
        transform_type = features.get("transform_type", "radial")
        n_copies = features.get("n_copies", 4)
        transformed = _apply_transform(base, transform_type, n_copies)
    else:
        transformed = base
    
    # Apply composition (D3)
    if n_dimensions >= 3:
        comp_type = features.get("composition_type", "none")
        comp_scale = features.get("comp_scale", 0.5)
        composed = _apply_composition(transformed, comp_type, comp_scale)
    else:
        composed = transformed
    
    # Scale to canvas
    canvas_scale = _calculate_canvas_scale(features, n_dimensions)
    return f"(T {composed} (M {canvas_scale:.3f} 0 0 0))"

# Import for labels
from tasks.base_task import Task as _BaseTask

def _add_red_label(image_path: str, label: str) -> None:
    """Add red numeric label."""
    _BaseTask._add_label_to_image(image_path, label, image_path)

def _save_trial_summary(tile_paths: List[str], oddball_idx: int, out_path: str) -> None:
    """Save trial summary grid."""
    imgs = []
    for i, p in enumerate(tile_paths, start=1):
        img = Image.open(p).convert("RGB")
        color = "red" if i == oddball_idx else "gray"
        w = 10
        bordered = Image.new("RGB", (img.width + 2*w, img.height + 2*w), color)
        bordered.paste(img, (w, w))
        imgs.append(bordered)
    
    w, h = imgs[0].size
    grid = Image.new("RGB", (3*w, 2*h), "white")
    pos = [(0,0),(w,0),(2*w,0),(0,h),(w,h),(2*w,h)]
    for im, xy in zip(imgs, pos):
        grid.paste(im, xy)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    grid.save(out_path)

def run_generate(outdir: str, n_trials: int, reference_variance: int,
                 n_dimensions: int, seed: int) -> None:
    """Generate trials with proper oddball control."""
    RNG.seed(seed)
    random.seed(seed)
    os.makedirs(os.path.join(outdir, "trials"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "summaries"), exist_ok=True)

    if n_dimensions == 1:
        abstractions = D1_ABS
        max_var = len(D1_ABS) - 1  # 0-1
    elif n_dimensions == 2:
        abstractions = D2_ABS
        max_var = len(D2_ABS) - 1  # 0-3
    else:
        abstractions = D3_ABS
        max_var = len(D3_ABS) - 1  # 0-5
    reference_variance = min(max(reference_variance, 0), max_var)

    rows = []
    for t_idx in range(n_trials):
        # Initialize ALL dimensions with defaults first
        base = {}
        for dim in VALUE_SPACE:
            base[dim] = VALUE_SPACE[dim][0]
        # Then randomize only the active dimensions
        for dim in abstractions:
            base[dim] = random.choice(VALUE_SPACE[dim])
        
        # Freeze controls for inactive dimensions
        frozen = freeze_controls(base, list(VALUE_SPACE.keys()), abstractions)
        
        # Pick oddball abstraction
        oddball_abstraction = random.choice(abstractions)
        
        # Pick which other dims will vary (reference_variance controls this)
        varying_dims = pick_varying_dims(abstractions, oddball_abstraction, reference_variance)

        # Create reference schedules for varying dims
        ref_schedules = {}
        for d in varying_dims:
            ref_schedules[d] = assign_ref_values_for_dim(VALUE_SPACE[d], base[d])

        # Generate 5 reference tiles
        ref_tiles = []
        for i in range(5):
            f = dict(base)
            f.update(frozen)
            for d in abstractions:
                if d in varying_dims:
                    f[d] = ref_schedules[d][i]
                else:
                    f[d] = base[d]
            ref_tiles.append({"features": f.copy()})

        # Create oddball
        from tasks.common_variance import choose_oddball_nonoddball_values, pick_oddball_value_excluding_refs
        oddf = dict(ref_tiles[0]["features"])
        patch = choose_oddball_nonoddball_values(ref_tiles, varying_dims, oddball_abstraction)
        for d, v in patch.items():
            oddf[d] = v
        ref_vals_set = {t["features"][oddball_abstraction] for t in ref_tiles}
        oddf[oddball_abstraction] = pick_oddball_value_excluding_refs(
            VALUE_SPACE[oddball_abstraction], ref_vals_set, oddf[oddball_abstraction]
        )

        # Place oddball randomly
        oddball_idx = RNG.randint(1, 6)
        tiles = []
        img_paths = []
        ref_iter = iter(ref_tiles)
        
        for pos in range(1, 7):
            if pos == oddball_idx:
                f = oddf
            else:
                f = next(ref_iter)["features"]
            
            out_png = os.path.join(outdir, "trials", f"trial={t_idx}_{pos}.png")
            program = _build_program(f, n_dimensions)
            render_program_to_png(program, PolygonalRenderer(), out_png, pad=5.5)
            
            # Add label
            _add_red_label(out_png, str(pos))
            
            tiles.append(dict(index=pos, image_path=out_png, features=f.copy()))
            img_paths.append(out_png)

        # Save summary
        summary_path = os.path.join(outdir, "summaries", f"summary=trial{t_idx}.png")
        _save_trial_summary(img_paths, oddball_idx, summary_path)

        # Validate and save
        val = validate_trial_tiles(tiles, oddball_idx, oddball_abstraction, abstractions, reference_variance)
        rows.append(dict(
            trial_idx=t_idx,
            summary_path=summary_path,
            oddball_idx=oddball_idx,
            oddball_abstraction=oddball_abstraction,
            reference_variance=reference_variance,
            n_dimensions=n_dimensions,
            abstraction_columns=abstractions,
            **val,
            tiles=tiles,
        ))
    
    write_trial_jsonl(outdir, rows)
    
    # Write CSV
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


class _BaseRegularPolygonsTask(Task):
    renderer = PolygonalRenderer()
    
    def __init__(self, output_dir: str, n_dimensions: int, reference_variance: int,
                 n_trials: int, seed: int = 1248, **kwargs):
        self.level = n_dimensions
        
        # Map to actual dims
        level_to_dims = {
            1: len(D1_ABS),  # 2
            2: len(D2_ABS),  # 4
            3: len(D3_ABS),  # 6
        }
        actual_dims = level_to_dims[self.level]
        
        tn = kwargs.pop("task_name", "regular_polygons")
        output_path = Path(output_dir)
        kwargs['data_dir'] = str(output_path.parent)
        kwargs['dataset_dir_override'] = str(output_path)
        kwargs['n_dimensions'] = actual_dims
        super().__init__(task_name=tn, **kwargs)
        
        self.output_dir = output_path
        self.n_dimensions = actual_dims
        self.reference_variance = reference_variance
        self.n_trials = n_trials
        self.seed = seed
        self.trials_metadata_path = self.output_dir / "trials.csv"
    
    def generate_programs(self) -> pd.DataFrame:
        """Generate all program combinations."""
        if self.level == 1:
            abstractions = D1_ABS
        elif self.level == 2:
            abstractions = D2_ABS
        else:
            abstractions = D3_ABS
        
        print(f"Generating programs for level {self.level} ({len(abstractions)} dimensions)")
        
        rows = []
        for combo in itertools.product(*[VALUE_SPACE[d] for d in abstractions]):
            features = {}
            for dim, val in zip(abstractions, combo):
                features[dim] = val
            
            program = _build_program(features, self.level)
            row = {"program_string": program}
            row.update(features)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        print(f"Generated {len(df)} programs")
        return df


class RegularPolygonsOneDimTask(_BaseRegularPolygonsTask):
    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "trials").mkdir(exist_ok=True)
        (self.output_dir / "summaries").mkdir(exist_ok=True)

        meta_path = self.output_dir / "metadata.csv"
        if self.overwrite_existing or (not meta_path.exists()):
            self._generate_and_render_stimuli()

        need_trials = self.overwrite_existing or (not self.trials_metadata_path.exists())
        if need_trials:
            run_generate(str(self.output_dir), self.n_trials, self.reference_variance, self.level, self.seed)

        meta_df = pd.read_csv(meta_path) if meta_path.exists() else pd.DataFrame()
        trials_df = pd.read_csv(self.trials_metadata_path) if self.trials_metadata_path.exists() else pd.DataFrame()
        return meta_df, trials_df


class RegularPolygonsTwoDimTask(_BaseRegularPolygonsTask):
    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "trials").mkdir(exist_ok=True)
        (self.output_dir / "summaries").mkdir(exist_ok=True)

        meta_path = self.output_dir / "metadata.csv"
        if self.overwrite_existing or (not meta_path.exists()):
            self._generate_and_render_stimuli()

        need_trials = self.overwrite_existing or (not self.trials_metadata_path.exists())
        if need_trials:
            run_generate(str(self.output_dir), self.n_trials, self.reference_variance, self.level, self.seed)

        meta_df = pd.read_csv(meta_path) if meta_path.exists() else pd.DataFrame()
        trials_df = pd.read_csv(self.trials_metadata_path) if self.trials_metadata_path.exists() else pd.DataFrame()
        return meta_df, trials_df


class RegularPolygonsThreeDimTask(_BaseRegularPolygonsTask):
    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "trials").mkdir(exist_ok=True)
        (self.output_dir / "summaries").mkdir(exist_ok=True)

        meta_path = self.output_dir / "metadata.csv"
        if self.overwrite_existing or (not meta_path.exists()):
            self._generate_and_render_stimuli()

        need_trials = self.overwrite_existing or (not self.trials_metadata_path.exists())
        if need_trials:
            run_generate(str(self.output_dir), self.n_trials, self.reference_variance, self.level, self.seed)

        meta_df = pd.read_csv(meta_path) if meta_path.exists() else pd.DataFrame()
        trials_df = pd.read_csv(self.trials_metadata_path) if self.trials_metadata_path.exists() else pd.DataFrame()
        return meta_df, trials_df
