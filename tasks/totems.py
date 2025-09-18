import pandas as pd
import math
import itertools
from tasks.base_task import Task
from renderer.languages.colored import ColoredRenderer


class TotemsTask(Task):
    """
    Generates totem-style stimuli composed of vertically stacked and colored polygons.
    """
    # --- specify the renderer for the task ---
    renderer = ColoredRenderer()

    def __init__(self, task_name=None, **kwargs):
        """Initializes the TotemsTask."""
        super().__init__(task_name=task_name or "totems", **kwargs)
        self.total_totem_height = 8.0
        self.stroke_width = 6.0

    # --- shape generation helpers ---
    def _generate_polygon_by_radius(self, n_sides, radius, y_offset=0.0):
        """Generates a "flat-top" polygon, vertically centered."""
        side_length = 2 * radius * math.sin(math.pi / n_sides)
        apothem = radius * math.cos(math.pi / n_sides)
        rotation_angle = (2 * math.pi) / n_sides
        horizontal_translation = -side_length / 2.0
        side_transform = f"(M {side_length} 0 {horizontal_translation} {apothem})"
        side_template = f"(T l {side_transform})"
        shape_str = f"(repeat {side_template} {n_sides} (M 1 {rotation_angle} 0 0))"
        if y_offset != 0.0:
            return f"(T {shape_str} (M 1 0 0 {y_offset}))"
        return shape_str

    def generate_shape_program(self, shape_type, height):
        """Generates a vertically centered, "flat-top" polygon of a precise height."""
        shape_map = {'circle': 60, 'triangle': 3, 'square': 4}
        n_sides = shape_map[shape_type]
        y_offset = 0.0
        if n_sides % 2 == 0:
            radius = height / (2 * math.cos(math.pi / n_sides))
        else:
            radius = height / (1 + math.cos(math.pi / n_sides))
            apothem = radius * math.cos(math.pi / n_sides)
            y_offset = (radius - apothem) / 2.0
        return self._generate_polygon_by_radius(n_sides, radius, y_offset)

    def _create_program_record(self, modules):
        """
        Generates a single totem program and its record from a tuple of
        (shape, color) pairs.
        """
        num_modules = len(modules)
        module_height = self.total_totem_height / num_modules
        color_map = {"red": "1 0 0", "green": "0 1 0", "blue": "0 0 1"}

        # generate all parts
        all_parts = []
        current_y = -self.total_totem_height / 2.0
        for shape_type, color_name in modules:
            y_pos = current_y + module_height / 2.0
            shape_geometry_str = self.generate_shape_program(shape_type, module_height)
            translated_shape = f"(T {shape_geometry_str} (M 1 0 0 {y_pos}))"
            module_str = f"(color {color_map[color_name]} {translated_shape})"
            all_parts.append(module_str)
            current_y += module_height

        # combine all parts into a single program string
        program_string = all_parts[0]
        for part in all_parts[1:]:
            program_string = f"(C {program_string} {part})"

        # create the record dictionary dynamically
        record = {}
        for i, (shape, color) in enumerate(modules):
            record[f"module_{i+1}_shape"] = shape
            record[f"module_{i+1}_color"] = color
        record["program_string"] = program_string
        return record

    def generate_programs(self):
        """Generates totems with a fixed number of 3 modules and color options."""
        # define the hyperparameter grid
        num_modules = 3
        shapes = ['circle', 'triangle', 'square']
        colors = ['red', 'green', 'blue']
        module_options = list(itertools.product(shapes, colors))
        all_combinations = itertools.product(module_options, repeat=num_modules)

        # generate programs for all combinations of parameters
        records = [self._create_program_record(modules) for modules in all_combinations]
        print(f"✅ Generated {len(records)} total unique programs.")
        return pd.DataFrame(records)


if __name__ == "__main__":
    task = TotemsTask()
    task.run()