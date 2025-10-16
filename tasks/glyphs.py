import pandas as pd
import numpy as np
import math
import itertools
from tasks.base_task import Task
from renderer.languages.probabilistic import ProbabilisticRenderer


class GlyphsTask(Task):
    """
    A program generation task for creating glyph stimuli composed of
    a base shape and up to three embellishments.
    """
    renderer = ProbabilisticRenderer()

    def __init__(self, task_name=None, dup_factor: int = 1, **kwargs):
        super().__init__(task_name=task_name or "glyphs", **kwargs)
        self._define_program_primitives()
        self.colors = ['red', 'green', 'blue']
        self.dup_factor = max(1, int(dup_factor))

    def _define_program_primitives(self):
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

        dot_radius = 0.1
        semicircle_length = math.pi * dot_radius
        dot_bend = 1.0
        single_dot_prog = f'(curve pi {semicircle_length:.3f} {dot_bend}) (curve pi {semicircle_length:.3f} {dot_bend})'

        # add 'none' → renders nothing when selected
        self.embellishments = {
            'none': None,
            'h_line':   '(C (lift -1.5 {y_pos} 0) (curve 0 3.0 0))',
            's_curve':  '(C (lift -1.5 {y_pos} (/ pi 7.25)) (curve -0.8 1.5 0.5) (curve 0.8 1.5 0.5))',
            'two_dots': f'(C (lift -0.5 {{y_pos}} (/ pi 2)) {single_dot_prog} (lift 0.5 {{y_pos}} (/ pi 2)) {single_dot_prog})'
        }
        self.emb_locations = {'top': -2.0, 'middle': 0.0, 'bottom': 2.0}

    def _create_program_record(self, params):
        (
            name_left,
            name_right,
            top_name,
            mid_name,
            bot_name,
            top_color,
            mid_color,
            bot_color,
        ) = params

        # 1) base
        prog_left = self.base_programs[name_left]
        prog_right = self.base_programs[name_right]
        base_part = f'(C {self.left_lift} {prog_left} {self.right_lift} {prog_right})'

        # 2) embellishments (skip any set to 'none')
        color_map = {"red": "1 0 0", "green": "0 1 0", "blue": "0 0 1"}
        parts = []

        if top_name != 'none':
            top_prog = self.embellishments[top_name].format(y_pos=self.emb_locations['top'])
            parts.append(f"(color {color_map[top_color]} {top_prog})")
        if mid_name != 'none':
            mid_prog = self.embellishments[mid_name].format(y_pos=self.emb_locations['middle'])
            parts.append(f"(color {color_map[mid_color]} {mid_prog})")
        if bot_name != 'none':
            bot_prog = self.embellishments[bot_name].format(y_pos=self.emb_locations['bottom'])
            parts.append(f"(color {color_map[bot_color]} {bot_prog})")

        embellishment_part = parts[0] if parts else None
        for p in parts[1:]:
            embellishment_part = f'(C {embellishment_part} {p})'

        program_string = f'(C {base_part} {embellishment_part})' if embellishment_part else base_part

        return {
            "base_left": name_left,
            "base_right": name_right,
            "embellishment_top": top_name,
            "embellishment_middle": mid_name,
            "embellishment_bottom": bot_name,
            "top_color": top_color,
            "middle_color": mid_color,
            "bottom_color": bot_color,
            "program_string": program_string,
        }

    def generate_programs(self):
        # not used by variants path
        return pd.DataFrame([])


if __name__ == "__main__":
    task = GlyphsTask()
    task.run()
