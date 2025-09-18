# dsl_renderer/colored.py
"""
Colored deterministic DSL renderer extending geometric operations with color support.

This module extends the deterministic DSL with color capabilities while maintaining
all the geometric operations and deterministic behavior. It modifies the stroke format
to include color information and provides a color application function.

The colored DSL supports:
- All deterministic DSL operations (T, C, repeat, M, primitives, math)
- Color application: color command to apply RGB values to strokes
- Colored stroke format: (stroke_array, (r, g, b)) tuples

Example programs:
    "(color 1 0 0 l)"  # Red line
    "(C (color 0 1 0 l) (color 0 0 1 c))"  # Green line + blue circle
    "(color 0.5 0.5 1 (repeat (T l 1 0.5) 8 (M 1 0.785)))"  # Light blue star
"""
import math
import numpy as np
from .deterministic import DeterministicRenderer, _line, _circle, _rectangle
from .deterministic import _apply_transform as _apply_transform_geom


# --- Helper functions modified for color ---
def _apply_transform_color(strokes, matrix):
    """
    Applies geometric transformation to colored strokes while preserving color information.
    
    Takes a list of colored strokes (stroke_array, color) tuples and applies the
    transformation matrix to each stroke's geometry while keeping the color unchanged.
    
    Args:
        strokes: List of (stroke_array, (r, g, b)) tuples
        matrix: 3x3 affine transformation matrix
        
    Returns:
        List of transformed colored strokes with preserved colors
        
    Examples:
        >>> colored_line = [(_line[0], (1.0, 0.0, 0.0))]  # Red line
        >>> transform = _make_affine_matrix(s=2.0)
        >>> _apply_transform_color(colored_line, transform)
        # Returns red line scaled by 2
    """
    if not strokes:
        return []
    return [(_apply_transform_geom(s, matrix), color) for s, color in strokes]


def _repeat_color(stroke, n, matrix):
    """
    Repeats colored strokes multiple times with cumulative transformations.
    
    Creates n copies of the colored stroke list, where each successive copy has the
    transformation matrix applied cumulatively to the geometry while preserving
    the original colors.
    
    Args:
        stroke: List of (stroke_array, (r, g, b)) tuples to repeat
        n: Number of repetitions (converted to int)
        matrix: 3x3 transformation matrix applied cumulatively to geometry
        
    Returns:
        List of all repeated colored strokes
        
    Examples:
        >>> red_line = [(_line[0], (1.0, 0.0, 0.0))]
        >>> rotate_matrix = _make_affine_matrix(theta=math.pi/4)
        >>> _repeat_color(red_line, 8, rotate_matrix)
        # Creates 8 red lines rotated by 45° increments
    """
    strokes = []
    current_stroke = stroke
    for i in range(int(n)):
        if i > 0:
            current_stroke = _apply_transform_color(current_stroke, matrix)
        strokes.extend(current_stroke)
    return strokes

def _color(r, g, b, strokes):
    """
    Applies a specified RGB color to a list of strokes.
    
    Takes stroke data (which may or may not already have color information) and
    applies a new RGB color to all strokes in the list. This function converts
    plain stroke arrays to colored stroke tuples.
    
    Args:
        r: Red component (0.0 to 1.0)
        g: Green component (0.0 to 1.0)  
        b: Blue component (0.0 to 1.0)
        strokes: List of stroke arrays or (stroke_array, color) tuples
        
    Returns:
        List of (stroke_array, (r, g, b)) tuples with the specified color
        
    Examples:
        >>> plain_strokes = [_line[0], _circle[0]]
        >>> _color(1.0, 0.0, 0.0, plain_strokes)
        # Returns red colored versions of line and circle
        >>> _color(0.0, 1.0, 0.0, [(stroke, (1, 0, 0))])  
        # Changes red stroke to green
    """
    new_color = (float(r), float(g), float(b))
    return [(s, new_color) for s, _ in strokes]


class ColoredRenderer(DeterministicRenderer):
    """
    Colored variant of the deterministic geometric DSL renderer.
    
    Extends the DeterministicRenderer with color support while maintaining all
    the geometric operations and deterministic behavior. All primitives and 
    operations now work with colored stroke tuples, and a new 'color' command
    is available for applying colors to stroke lists.
    
    The colored stroke format uses (stroke_array, (r, g, b)) tuples where:
    - stroke_array: numpy array of 2D points
    - (r, g, b): RGB color tuple with values in [0.0, 1.0] range
    
    Additional DSL Operations:
        color r g b strokes: Apply RGB color to stroke list
        
    All other operations work identically to DeterministicRenderer but preserve colors.
    
    Examples:
        >>> renderer = ColoredRenderer()
        >>> ast = parse_program("(color 1 0 0 (T l 2 0))")  # Red scaled line
        >>> strokes = renderer.evaluate(ast)
        >>> strokes[0][1]  # Color tuple
        (1.0, 0.0, 0.0)
    """
    
    def __init__(self):
        super().__init__()
        self._register_dsl_specific()

    def _register_dsl_specific(self):
        """
        Registers primitives and functions for the colored DSL.
        
        Overrides the base deterministic DSL to:
        1. Convert basic geometric primitives to colored format with default black color
        2. Replace transformation functions with color-preserving versions  
        3. Add the new 'color' command for applying colors to strokes
        
        All geometric operations preserve color information through transformations.
        """
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