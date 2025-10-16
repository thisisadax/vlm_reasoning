import pandas as pd
import numpy as np
import math
import itertools  # Import the itertools library
from tasks.base_task import Task
from renderer.languages.deterministic import DeterministicRenderer


class SpirographsTask(Task):
    """
    A program generation task for creating spirograph-style or mandala-like images.
    """
    # --- specify the renderer for the task ---
    renderer = DeterministicRenderer()

    def __init__(
        self,
        task_name=None,
        dimension_mode: str = "both",
        dup_factor: int = 1,
        exclude_abstraction_keywords=None,  # <-- NEW: accept and forward
        **kwargs
    ):
        """Initializes the SpirographsTask.

        dimension_mode controls which abstraction groups are present in rendering and metadata.
        Supported values:
          - "both": central + radial groups (default; legacy behavior)
          - "central": only central primitive/scale present; no radial geometry
          - "radial": only radial primitive/scale/radius/n_repeats present; no central geometry
        """
        self.dimension_mode = dimension_mode  # both | central | radial
        self.dup_factor = max(1, int(dup_factor))
        super().__init__(
            task_name=task_name or "spirographs",
            dup_factor=dup_factor,
            exclude_abstraction_keywords=exclude_abstraction_keywords,  # <-- pass through
            **kwargs,
        )

    # --- shape generation helpers ---
    def generate_polygon(self, n_sides, scale):
        """Generates a "pointed" regular n-sided polygon."""
        side_length = 2 * scale * math.sin(math.pi / n_sides)
        apothem = scale * math.cos(math.pi / n_sides)
        rotation_angle = (2 * math.pi) / n_sides
        horizontal_translation = -side_length / 2.0
        side_transform = f"(M {side_length} 0 {horizontal_translation} {apothem})"
        side_template = f"(T l {side_transform})"
        flat_top_polygon = f"(repeat {side_template} {n_sides} (M 1 {rotation_angle} 0 0))"
        initial_rotation = math.pi / n_sides
        return f"(T {flat_top_polygon} (M 1 {initial_rotation} 0 0))"

    def generate_shape_program(self, primitive, scale, is_radial=False):
        """Generates a program string for a shape with a normalized size."""
        polygon_map = {'c': 60, 't': 3, 's': 4, 'p': 5, 'h': 6}
        if primitive in polygon_map:
            n_sides = polygon_map[primitive]
            shape_str = self.generate_polygon(n_sides, scale)
            if is_radial:
                rotation_angle_rads = -math.pi / 2
                return f"(T {shape_str} (M 1 {rotation_angle_rads} 0 0))"
            else:
                return shape_str
        return ""

    def _create_program_record(self, params, n_dimensions):
        """
        Generates a single program string and its corresponding record
        from a tuple of parameters, based on n_dimensions.
        """
        # Build program and record based on dimension_mode and n_dimensions
        record = {}
        program_parts = []

        if self.dimension_mode in ("both", "central"):
            # Central group
            c_prim = params[0]
            record["central_primitive"] = c_prim
            if n_dimensions == 1 and self.dimension_mode == "central":
                # widened fallback central size
                central_part = self.generate_shape_program(c_prim, 1.5, is_radial=False)
                program_parts.append(central_part)
            elif self.dimension_mode == "both" and n_dimensions == 1:
                # In both-mode with n_dimensions == 1, treat as central-only primitive
                # widened fallback central size
                central_part = self.generate_shape_program(c_prim, 1.5, is_radial=False)
                program_parts.append(central_part)
            else:
                # central scale exists for n_dimensions >= 2 in central or both mode
                c_scale_idx = 1
                c_scale = params[c_scale_idx] if n_dimensions >= 2 else 1.5  # widened fallback
                if n_dimensions >= 2:
                    record["central_scale"] = c_scale
                central_part = self.generate_shape_program(c_prim, c_scale, is_radial=False)
                program_parts.append(central_part)

        # Radial group
        add_radial = self.dimension_mode in ("both", "radial") and (
            (self.dimension_mode == "both" and n_dimensions >= 3) or (self.dimension_mode == "radial")
        )

        if add_radial:
            # Determine base index in params depending on whether central group consumed slots
            if self.dimension_mode == "both":
                base = 2  # after central_primitive, central_scale
                dims_available = n_dimensions - 2
            elif self.dimension_mode == "radial":
                base = 0
                dims_available = n_dimensions
            else:
                base = 0
                dims_available = 0

            # Map dims to radial parameters
            # dims_available: 1-> r_prim; 2-> + r_scale; 3-> + radius; 4-> + n_repeats
            r_prim = params[base + 0] if dims_available >= 1 else 'c'
            if dims_available >= 1:
                record["radial_primitive"] = r_prim
            r_scale = params[base + 1] if dims_available >= 2 else 0.85  # widened fallback
            if dims_available >= 2:
                record["radial_scale"] = r_scale
            radius = params[base + 2] if dims_available >= 3 else 3.0
            if dims_available >= 3:
                record["radius"] = radius
            n = params[base + 3] if dims_available >= 4 else 4
            if dims_available >= 4:
                record["n_repeats"] = n

            radial_base_shape = self.generate_shape_program(r_prim, r_scale, is_radial=True)
            radial_positioned_shape = f"(T {radial_base_shape} (M 1 0 {radius} 0))"
            angle = (2 * math.pi) / n
            radial_part = f"(repeat {radial_positioned_shape} {n} (M 1 {angle} 0 0))"
            program_parts.append(radial_part)

        # Combine parts
        if len(program_parts) == 0:
            program_string = ""
        elif len(program_parts) == 1:
            program_string = program_parts[0]
        else:
            program_string = f"(C {program_parts[0]} {program_parts[1]})"

        record["program_string"] = program_string
        return record

    def generate_programs(self):
        """Generates a DataFrame of spirograph-style programs using itertools."""

        # Define the full hyperparameter grid by mode
        central_grid = [
            ['c', 's', 't', 'p', 'h'],        # central_primitives
            # WIDENED central scales (bigger separation, same bounds)
            [1.0, 1.5, 2.0, 2.5, 3.0],        # central_scales
        ]
        radial_grid = [
            ['c', 's', 't', 'p', 'h'],        # radial_primitives
            # WIDENED radial scales (still canvas-safe)
            [0.45, 0.65, 0.85, 1.05, 1.25],   # radial_scales
            [2.5, 3.0, 3.5, 4.0, 4.5],        # radii
            [3, 4, 5, 6, 7],                   # n_repeats
        ]

        if self.dimension_mode == "both":
            full_param_grid = central_grid + radial_grid
            param_grid = full_param_grid[:self.n_dimensions]
        elif self.dimension_mode == "central":
            full_param_grid = central_grid
            # Cap n_dimensions to [1,2] for central-only
            dims = max(1, min(self.n_dimensions, 2))
            param_grid = full_param_grid[:dims]
        elif self.dimension_mode == "radial":
            full_param_grid = radial_grid
            # Cap n_dimensions to [1..4] for radial-only
            dims = max(1, min(self.n_dimensions, 4))
            param_grid = full_param_grid[:dims]
        else:
            raise ValueError(f"Unknown dimension_mode: {self.dimension_mode}")

        # Feasibility filter to make radius and central size more obvious and avoid overlap/off-canvas
        def combo_ok(params_tuple):
            canvas_bound = 5.0
            if self.dimension_mode == "both":
                c_scale = params_tuple[1] if len(params_tuple) >= 2 else 1.5   # widened fallback
                r_scale = params_tuple[3] if len(params_tuple) >= 4 else 0.85  # widened fallback
                radius  = params_tuple[4] if len(params_tuple) >= 5 else 3.0
            elif self.dimension_mode == "central":
                c_scale = params_tuple[1] if len(params_tuple) >= 2 else 1.5   # widened fallback
                r_scale = 0.0
                radius  = 0.0
            else:  # radial
                c_scale = 0.0
                r_scale = params_tuple[1] if len(params_tuple) >= 2 else 0.85  # widened fallback
                radius  = params_tuple[2] if len(params_tuple) >= 3 else 3.0

            # Ensure ring outside central shape and inside canvas
            if radius:
                if radius < (c_scale + 0.7 * r_scale):
                    return False
                if (radius + 0.8 * r_scale) > (canvas_bound - 0.2):
                    return False
            # Central stays within canvas comfortably
            if (c_scale + 0.5) > (canvas_bound - 0.2):
                return False
            return True

        all_combinations = (p for p in itertools.product(*param_grid) if combo_ok(p))

        # generate programs for all combinations of parameters
        records = [self._create_program_record(params, self.n_dimensions) for params in all_combinations]
        #if self.dup_factor > 1:
            # replicate records to create per-abstraction duplicates for sampling feasibility
            #records = [rec for rec in records for _ in range(self.dup_factor)]
        print(f"✅ Generated {len(records)} total unique programs for n_dimensions={self.n_dimensions}.")
        return pd.DataFrame(records)


if __name__ == "__main__":
    task = SpirographsTask()
    task.run()
