import pandas as pd
import numpy as np
import math
import itertools
from tasks.base_task import Task
from renderer.languages.probabilistic import ProbabilisticRenderer


class GlyphsTask(Task):
    """
    A program generation task for creating glyph stimuli composed of
    a base shape and three embellishments.
    """
    # --- specify the renderer for the task ---
    renderer = ProbabilisticRenderer()

    def __init__(self, task_name=None, **kwargs):
        """Initializes the GlyphsTask."""
        super().__init__(task_name=task_name or "glyphs", **kwargs)
        self._define_program_primitives()

    def _define_program_primitives(self):
        """Defines the DSL program parts as instance attributes."""
        # --- Base Shape Parameters ---
        total_height = 4.0
        d_angle = 2 * math.atan(1.0 / 4.0)
        sweep_bend = 0.5

        self.base_programs = {
            'vertical': f'(curve 0 {total_height} 0)',
            'slant_left': f'(curve {d_angle:.4f} {total_height} 0)',
            'slant_right': f'(curve {-d_angle:.4f} {total_height} 0)',
            'sweep_left': f'(curve {d_angle:.4f} {total_height} {sweep_bend})',
            'sweep_right': f'(curve {-d_angle:.4f} {total_height} {sweep_bend})'
        }
        self.left_lift = f'(lift -1 {-total_height/2.0} (/ pi 2))'
        self.right_lift = f'(lift 1 {-total_height/2.0} (/ pi 2))'

        # --- Embellishment Parameters ---
        dot_radius = 0.1
        semicircle_length = math.pi * dot_radius
        dot_bend = 1.0
        single_dot_prog = f'(curve pi {semicircle_length:.3f} {dot_bend}) (curve pi {semicircle_length:.3f} {dot_bend})'

        self.embellishments = {
            'h_line':   '(C (lift -1.5 {y_pos} 0) (curve 0 3.0 0))',
            's_curve':  '(C (lift -1.5 {y_pos} (/ pi 7.25)) (curve -0.8 1.5 0.5) (curve 0.8 1.5 0.5))',
            'two_dots': f'(C (lift -0.5 {{y_pos}} (/ pi 2)) {single_dot_prog} (lift 0.5 {{y_pos}} (/ pi 2)) {single_dot_prog})'
        }
        self.emb_locations = {'top': -2.0, 'middle': 0.0, 'bottom': 2.0}

    def _create_program_record(self, params):
        """
        Generates a single program string and its corresponding record
        from a tuple of parameters.
        """
        (name_left, name_right), (top_name, mid_name, bot_name) = params

        # 1. Construct the base shape program part
        prog_left = self.base_programs[name_left]
        prog_right = self.base_programs[name_right]
        base_part = f'(C {self.left_lift} {prog_left} {self.right_lift} {prog_right})'

        # 2. Construct the positioned embellishments program part
        top_prog = self.embellishments[top_name].format(y_pos=self.emb_locations['top'])
        mid_prog = self.embellishments[mid_name].format(y_pos=self.emb_locations['middle'])
        bot_prog = self.embellishments[bot_name].format(y_pos=self.emb_locations['bottom'])
        embellishment_part = f'(C {top_prog} {mid_prog} {bot_prog})'

        # 3. Combine base and embellishments into the final program
        program_string = f'(C {base_part} {embellishment_part})'

        return {
            "base_left": name_left,
            "base_right": name_right,
            "embellishment_top": top_name,
            "embellishment_middle": mid_name,
            "embellishment_bottom": bot_name,
            "program_string": program_string,
        }

    def generate_programs(self):
        """Generates a DataFrame of glyph programs using itertools."""

        # generate all combinations for base shapes (left/right) and embellishments (top/middle/bottom)
        base_names = list(self.base_programs.keys())
        emb_names = list(self.embellishments.keys())
        base_combinations = list(itertools.product(base_names, repeat=2))
        embellishment_combinations = list(itertools.product(emb_names, repeat=3))
        all_combinations = list(itertools.product(base_combinations, embellishment_combinations))
        records = [self._create_program_record(params) for params in all_combinations]

        # generate programs for all combinations of parameters.
        num_base = len(base_combinations)
        num_emb = len(embellishment_combinations)
        print(f"✅ Generated {len(records)} total unique programs ({num_base} base shapes × {num_emb} embellishment sets).")
        return pd.DataFrame(records)


if __name__ == "__main__":
    task = GlyphsTask()
    task.run()