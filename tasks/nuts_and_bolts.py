import pandas as pd
import numpy as np
import math
import itertools
from tasks.base_task import Task
from renderer.languages.deterministic import DeterministicRenderer


class NutsAndBoltsTask(Task):
    """
    A program generation task for creating "nuts 'n bolts"-style images.
    """
    # --- Specify the renderer for the task ---
    renderer = DeterministicRenderer()

    def __init__(self, task_name=None, **kwargs):
        """Initializes the NutsAndBoltsTask."""
        super().__init__(task_name=task_name or "nuts_and_bolts", **kwargs)
        self.polygon_map = {'c': 60, 't': 3, 's': 4, 'p': 5, 'h': 6}
        self.canvas_bound = 5.0

    # --- Shape generation helpers ---
    def generate_polygon(self, n_sides, scale):
        """Generates a "pointy-top" regular n-sided polygon."""
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
        """Generates a program string for any shape."""
        if primitive in self.polygon_map:
            n_sides = self.polygon_map[primitive]
            shape_str = self.generate_polygon(n_sides, scale)
            if is_radial:
                rotation_angle_rads = -math.pi / 2
                return f"(T {shape_str} (M 1 {rotation_angle_rads} 0 0))"
            return shape_str
        return ""

    def generate_inner_ring(self, shape_type, n_shapes, radius):
        """Generates the program string for a radially repeating ring of shapes."""
        shape_scale = 0.5
        base_shape = self.generate_shape_program(shape_type, shape_scale, is_radial=True)
        positioned_shape = f"(T {base_shape} (M 1 0 {radius} 0))"
        rotation_angle = (2 * math.pi) / n_shapes
        return f"(repeat {positioned_shape} {n_shapes} (M 1 {rotation_angle} 0 0))"

    def _create_program_record(self, params):
        """
        Generates a single program and its record from a tuple of parameters,
        including dynamic geometry calculations.
        """
        os1_type, os2_type, os3_scale, is_type, is_n = params
        outer_scale_large = self.canvas_bound - 0.25
        outer_scale_small = self.canvas_bound - 0.5
        
        R_inner_central = os3_scale
        R_radial_shape = 0.5
        n_sides_outer = self.polygon_map[os1_type]
        apothem_outer = outer_scale_small * math.cos(math.pi / n_sides_outer)
        
        min_radius = R_inner_central + R_radial_shape
        max_radius = apothem_outer - R_radial_shape
        valid_radius = (min_radius + max_radius) / 2.0

        part1 = self.generate_shape_program(os1_type, outer_scale_small)
        part2 = self.generate_shape_program(os1_type, outer_scale_large)
        part3 = self.generate_shape_program(os2_type, os3_scale)
        base_program = f"(C (C {part1} {part2}) {part3})"
        
        inner_part = self.generate_inner_ring(is_type, is_n, valid_radius)
        program_string = f"(C {base_program} {inner_part})"

        return {
            "outer_shape_type": os1_type, 
            "nested_shape_type": os2_type,
            "nested_shape_scale": os3_scale, 
            "inner_shape_type": is_type,
            "inner_n_shapes": is_n, 
            "program_string": program_string
        }

    def generate_programs(self):
        """Generates a DataFrame of programs using an itertools-based approach."""
        
        # define the hyperparameter grid
        param_grid = [
            ['c', 's', 'p', 'h'],      # outer_shape_types
            ['c', 's', 'p', 'h'],      # nested_shape_types
            [1.0, 1.5, 2.0],           # third_shape_scales
            ['c', 's', 'p', 'h'],      # inner_shape_types
            [3, 4, 5, 6]               # inner_n_shapes
        ]
        all_combinations = itertools.product(*param_grid)
        
        # generate programs for all combinations of parameters.
        records = [self._create_program_record(p) for p in all_combinations]
        print(f"✅ Generated {len(records)} total valid programs.")
        return pd.DataFrame(records)


if __name__ == "__main__":
    task = NutsAndBoltsTask()
    task.run()