# dsl_renderer/polygonal.py
"""
Polygonal geometric DSL renderer for chained geometric constructions.

This module implements a stateless, deterministic DSL for creating drawings
primarily based on lines and arcs. It features a powerful 'repeat' operator
that enables chained transformations, where each successive shape begins at the
endpoint of the previous one, allowing for easy generation of polygons, spirals,
and other complex sequential patterns.

The DSL vocabulary includes:
- Primitives: l (line)
- Functions: a (arc), T (transform), C (compose), repeat, M (affine matrix)
- Math: sin, cos, tan, +, -, *, /, pi

Example programs:
    "(repeat l 4 0 0 1 1.5708)"  # Creates a square
    "(repeat (a 0.2) 6 0 0 1 1.0472)" # Creates a hexagon with concave sides
"""
import math
import numpy as np
from ..core import AstNode 
from .base import BaseRenderer 

## --- Geometric Primitives & Values ---
_line = [np.array([(0.0, 0.0), (1.0, 0.0)])]  # Unit line from origin to (1,0)

def _create_arc(curviness, num_points=30):
    """
    Generates a single stroke for a curved arc using a quadratic Bezier curve.

    The arc is drawn from (0,0) to (1,0). The 'curviness' parameter controls
    the y-coordinate of the Bezier control point, which is set at x=0.5.
    A positive value creates a concave arc, a negative value creates a convex one.

    Args:
        curviness (float): Determines the arc's curvature.
        num_points (int): The number of points to sample for the curve.

    Returns:
        list: A list containing a single numpy array of points for the stroke.
    """
    p0 = np.array([0.0, 0.0])
    p1 = np.array([0.5, curviness])  # The control point that defines the curve
    p2 = np.array([1.0, 0.0])

    t_values = np.linspace(0, 1, num_points)
    # Quadratic Bezier formula: B(t) = (1-t)^2*P0 + 2(1-t)t*P1 + t^2*P2
    points = np.outer((1 - t_values)**2, p0) + \
             np.outer(2 * (1 - t_values) * t_values, p1) + \
             np.outer(t_values**2, p2)
    return [points]

## --- Geometric Transformation Functions ---
def _make_affine_matrix(s=1.0, theta=0.0, x=0.0, y=0.0, order='trs'):
    """Creates a 3x3 affine transformation matrix."""
    s = s if s is not None else 1.0
    theta = theta if theta is not None else 0.0
    x = x if x is not None else 0.0
    y = y if y is not None else 0.0
    rotation = np.array([[math.cos(theta), -math.sin(theta), 0.0],
                         [math.sin(theta), math.cos(theta), 0.0], [0.0, 0.0, 1.0]])
    scale = np.array([[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, 1.0]])
    translation = np.array([[1.0, 0.0, x], [0.0, 1.0, y], [0.0, 0.0, 1.0]])
    op_map = {'t': translation, 'r': rotation, 's': scale}
    return op_map[order[0]] @ op_map[order[1]] @ op_map[order[2]]

def _apply_transform(strokes, matrix):
    """Applies an affine transformation matrix to strokes."""
    if isinstance(strokes, list):
        return [_apply_transform(s, matrix) for s in strokes]
    points_h = np.hstack([strokes, np.ones((strokes.shape[0], 1))])
    transformed_points = (matrix @ points_h.T).T
    return transformed_points[:, :2]

def _repeat_chained(stroke, n_repeats, tx, ty, s, theta):
    """
    Repeats a stroke with chained, cumulative transformations.

    Each repetition starts relative to the endpoint of the previous one. This
    is ideal for creating polygons, spirals, and other sequential patterns.

    Args:
        stroke (list): List of stroke arrays to repeat (e.g., a line or arc).
        n_repeats (float): Number of repetitions (will be cast to int).
        tx (float): Translation in x applied at each step.
        ty (float): Translation in y applied at each step.
        s (float): Scaling factor applied at each step.
        theta (float): Rotation in radians applied at each step.

    Returns:
        list: A list of stroke arrays for all repetitions.
    """
    strokes = []
    # Transformation to move from one primitive's local frame to the next.
    # 1. First, translate to the endpoint of the base primitive, which is (1,0).
    # 2. Then, apply the user-specified scaling, rotation, and translation.
    translate_to_end = _make_affine_matrix(x=1.0)
    user_transform = _make_affine_matrix(s=s, theta=theta, x=tx, y=ty)
    step_transform = translate_to_end @ user_transform

    current_transform = np.identity(3)  # Start with the identity matrix
    for _ in range(int(n_repeats)):
        # Apply the cumulative transform to the base stroke
        transformed_stroke = _apply_transform(stroke, current_transform)
        strokes.extend([np.copy(s) for s in transformed_stroke])
        # Update the transform for the next iteration by chaining the step
        current_transform = current_transform @ step_transform
    return strokes


class PolygonalRenderer(BaseRenderer):
    """
    Renderer for the stateless, polygonal geometry DSL.

    This renderer implements a DSL focused on lines and arcs with a powerful
    chained 'repeat' operation for building complex polygonal and spiral shapes.
    """
    def __init__(self):
        super().__init__()
        self._register_dsl_specific()

    def _register_dsl_specific(self):
        """Registers the primitives and functions for the polygonal DSL."""
        self.primitives.update({"l": _line})
        self.implementations.update({
            "a": _create_arc, "M": _make_affine_matrix, "T": _apply_transform,
            "C": lambda s1, s2: s1 + s2, "repeat": _repeat_chained,
            "tan": math.tan, "cos": math.cos, "sin": math.sin,
        })

    def evaluate(self, node: AstNode | float | int):
        """Evaluates an AST node to produce geometric strokes (stateless)."""
        if not isinstance(node, AstNode):
            return node
        if node.name in self.primitives:
            return self.primitives[node.name]
        evaluated_args = [self.evaluate(arg) for arg in node.args]
        if node.name in self.implementations:
            func = self.implementations[node.name]
            return func(*evaluated_args)
        raise ValueError(f"Unknown function or primitive: {node.name}")