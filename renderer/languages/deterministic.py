# dsl_renderer/deterministic.py
import math
import numpy as np
from ..core import AstNode
from .base import BaseRenderer


## --- Geometric Primitives & Values ---
_line = [np.array([(0.0, 0.0), (1.0, 0.0)])]
_circle = [np.array([(0.5 * math.cos(t), 0.5 * math.sin(t))
                     for t in np.linspace(0.0, 2.0 * math.pi, num=30)])]
_rectangle = [np.array([(-0.5, -0.5), (0.5, -0.5),
                         (0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5)])]


## --- Geometric Transformation Functions ---
def _make_affine_matrix(s=1.0, theta=0.0, x=0.0, y=0.0, order='trs'):
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
    if isinstance(strokes, list):
        return [_apply_transform(s, matrix) for s in strokes]
    points_h = np.hstack([strokes, np.ones((strokes.shape[0], 1))])
    transformed_points = (matrix @ points_h.T).T
    return transformed_points[:, :2]


def _repeat(stroke, n, matrix):
    strokes = []
    current_stroke = stroke
    for i in range(int(n)):
        if i > 0:
            current_stroke = _apply_transform(current_stroke, matrix)
        strokes.extend([np.copy(s) for s in current_stroke])
    return strokes


class DeterministicRenderer(BaseRenderer):
    """Renderer for the stateless, deterministic geometry DSL."""
    def __init__(self):
        super().__init__()
        self._register_dsl_specific()

    def _register_dsl_specific(self):
        """Registers primitives and functions for this DSL."""
        self.primitives.update({
            "l": _line, "c": _circle, "r": _rectangle,
        })
        self.implementations.update({
            "M": _make_affine_matrix, "T": _apply_transform, "C": lambda s1, s2: s1 + s2,
            "repeat": _repeat, "tan": math.tan, "cos": math.cos, "sin": math.sin,
        })

    def evaluate(self, node: AstNode | float | int):
        """Recursively evaluates an AST node (stateless)."""
        if not isinstance(node, AstNode):
            return node
        if node.name in self.primitives:
            return self.primitives[node.name]
        evaluated_args = [self.evaluate(arg) for arg in node.args]
        if node.name in self.implementations:
            func = self.implementations[node.name]
            return func(*evaluated_args)
        raise ValueError(f"Unknown function or primitive: {node.name}")