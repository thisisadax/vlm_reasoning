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

    def __init__(self, task_name=None, **kwargs):
        """Initializes the SpirographsTask."""
        super().__init__(task_name=task_name or "spirographs", **kwargs)
    
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

    def _create_program_record(self, params):
        """
        Generates a single program string and its corresponding record
        from a tuple of parameters.
        """
        c_prim, c_scale, r_prim, r_scale, radius, n = params
        central_part = self.generate_shape_program(c_prim, c_scale, is_radial=False)
        radial_base_shape = self.generate_shape_program(r_prim, r_scale, is_radial=True)
        radial_positioned_shape = f"(T {radial_base_shape} (M 1 0 {radius} 0))"
        angle = (2 * math.pi) / n
        radial_part = f"(repeat {radial_positioned_shape} {n} (M 1 {angle} 0 0))"
        program_string = f"(C {central_part} {radial_part})"
        return {
            "central_primitive": c_prim,
            "central_scale": c_scale,
            "radial_primitive": r_prim,
            "radial_scale": r_scale,
            "radius": radius,
            "n_repeats": n,
            "program_string": program_string,
        }

    def generate_programs(self):
        """Generates a DataFrame of spirograph-style programs using itertools."""

        # define the hyperparameter grid
        param_grid = [
            ['c', 's', 't', 'p', 'h'],  # central_primitives
            [1.25, 2.0],                # central_scales
            ['c', 's', 't', 'p', 'h'],  # radial_primitives
            [0.5, 0.75],                # radial_scales
            [3.0, 4.0],                 # radii
            [4, 6, 8],                  # n_repeats
        ]
        all_combinations = itertools.product(*param_grid)

        # generate programs for all combinations of parameters
        records = [self._create_program_record(params) for params in all_combinations]
        print(f"✅ Generated {len(records)} total unique programs.")
        return pd.DataFrame(records)


if __name__ == "__main__":
    task = SpirographsTask()
    task.run()