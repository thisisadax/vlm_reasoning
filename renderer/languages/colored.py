# dsl_renderer/colored.py
import math
import numpy as np
from .deterministic import DeterministicRenderer, _line, _circle, _rectangle
from .deterministic import _apply_transform as _apply_transform_geom
from ..core import AstNode


# --- Helper functions modified for color ---
def _apply_transform_color(strokes, matrix):
    """Applies a transform, preserving color."""
    if not strokes:
        return []
    return [(_apply_transform_geom(s, matrix), color) for s, color in strokes]


def _repeat_color(stroke, n, matrix):
    """Repeats a stroke, preserving color."""
    strokes = []
    current_stroke = stroke
    for i in range(int(n)):
        if i > 0:
            current_stroke = _apply_transform_color(current_stroke, matrix)
        strokes.extend(current_stroke)
    return strokes

def _color(r, g, b, strokes):
    """Applies a new color to a list of strokes."""
    new_color = (float(r), float(g), float(b))
    return [(s, new_color) for s, _ in strokes]


class ColoredRenderer(DeterministicRenderer):
    """A deterministic renderer that supports a 'color' command."""
    
    def __init__(self):
        super().__init__()
        self._register_dsl_specific()

    def _register_dsl_specific(self):
        """Registers primitives and functions for the colored DSL."""
        super()._register_dsl_specific()

        default_color = (0.0, 0.0, 0.0)
        self.primitives.update({
            "l": [(_line[0], default_color)],
            "c": [(_circle[0], default_color)],
            "r": [(_rectangle[0], default_color)],
        })
        
        self.implementations.update({
            "T": _apply_transform_color,
            "repeat": _repeat_color,
            "color": _color,
        })