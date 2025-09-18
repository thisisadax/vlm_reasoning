# dsl_renderer/deterministic.py
"""
Deterministic geometric DSL renderer for precise geometric constructions.

This module implements a stateless, deterministic DSL for creating geometric 
drawings. It supports basic shapes (line, circle, rectangle), affine transformations
(translation, rotation, scaling), and composition operations. All operations are
deterministic and produce identical output for identical input.

The DSL vocabulary includes:
- Primitives: l (line), c (circle), r (rectangle)  
- Functions: T (transform), C (compose), repeat, M (affine matrix)
- Math: sin, cos, tan, +, -, *, /, pi

Example programs:
    "(T l 2 1.5)"  # Transform line with scale=2, rotation=1.5
    "(C l c)"      # Compose line and circle
    "(repeat (T l 1 0.5) 6 (M 1 0.5))"  # Repeat transformed line 6 times
"""
import math
import numpy as np
from ..core import AstNode
from .base import BaseRenderer


## --- Geometric Primitives & Values ---
# Basic geometric shapes as lists of stroke arrays
_line = [np.array([(0.0, 0.0), (1.0, 0.0)])]  # Unit line from origin to (1,0)
_circle = [np.array([(0.5 * math.cos(t), 0.5 * math.sin(t))  # Unit circle, radius 0.5
                     for t in np.linspace(0.0, 2.0 * math.pi, num=30)])]
_rectangle = [np.array([(-0.5, -0.5), (0.5, -0.5),  # Unit square centered at origin
                         (0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5)])]


## --- Geometric Transformation Functions ---
def _make_affine_matrix(s=1.0, theta=0.0, x=0.0, y=0.0, order='trs'):
    """
    Creates a 3x3 affine transformation matrix.
    
    Combines scale, rotation, and translation operations in the specified order.
    The default order 'trs' applies translation, then rotation, then scaling.
    
    Args:
        s: Scale factor (default 1.0)
        theta: Rotation angle in radians (default 0.0) 
        x: Translation in x direction (default 0.0)
        y: Translation in y direction (default 0.0)
        order: Order of operations as string, e.g. 'trs', 'rst' (default 'trs')
        
    Returns:
        3x3 numpy array representing the combined affine transformation
        
    Examples:
        >>> _make_affine_matrix(s=2.0, theta=math.pi/4, x=1.0, y=0.5)
        # Creates matrix for scale=2, rotate=45°, translate=(1,0.5)
    """
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
    """
    Applies an affine transformation matrix to strokes.
    
    Recursively transforms either a single stroke (numpy array) or a list of strokes.
    Uses homogeneous coordinates to apply the 3x3 transformation matrix.
    
    Args:
        strokes: Single stroke array or list of stroke arrays  
        matrix: 3x3 affine transformation matrix
        
    Returns:
        Transformed strokes in the same format as input
        
    Examples:
        >>> line = np.array([[0, 0], [1, 0]])
        >>> transform = _make_affine_matrix(s=2.0)  
        >>> _apply_transform(line, transform)
        # Returns line scaled by factor of 2
    """
    if isinstance(strokes, list):
        return [_apply_transform(s, matrix) for s in strokes]
    points_h = np.hstack([strokes, np.ones((strokes.shape[0], 1))])
    transformed_points = (matrix @ points_h.T).T
    return transformed_points[:, :2]


def _repeat(stroke, n, matrix):
    """
    Repeats a stroke multiple times with cumulative transformations.
    
    Creates n copies of the stroke, where each successive copy has the 
    transformation matrix applied cumulatively. This allows for spiral
    patterns, scaling sequences, etc.
    
    Args:
        stroke: List of stroke arrays to repeat
        n: Number of repetitions (converted to int)
        matrix: 3x3 transformation matrix applied cumulatively
        
    Returns:
        List of stroke arrays representing all repetitions
        
    Examples:
        >>> line = [np.array([[0, 0], [1, 0]])]
        >>> rotate_matrix = _make_affine_matrix(theta=math.pi/3)
        >>> _repeat(line, 6, rotate_matrix)
        # Creates 6 lines rotated by 60° increments (hexagon pattern)
    """
    strokes = []
    current_stroke = stroke
    for i in range(int(n)):
        if i > 0:
            current_stroke = _apply_transform(current_stroke, matrix)
        strokes.extend([np.copy(s) for s in current_stroke])
    return strokes


class DeterministicRenderer(BaseRenderer):
    """
    Renderer for the stateless, deterministic geometry DSL.
    
    Implements a deterministic DSL for geometric constructions with basic shapes,
    affine transformations, and composition operations. All operations are pure
    functions that produce identical output for identical input.
    
    DSL Operations:
        Primitives: l, c, r (line, circle, rectangle)
        Functions: T (transform), C (compose), repeat, M (make matrix)
        Math: sin, cos, tan, +, -, *, /, pi
        
    Example Usage:
        >>> renderer = DeterministicRenderer()
        >>> ast = parse_program("(T l 2 1.5)")
        >>> strokes = renderer.evaluate(ast)  # Transformed line
    """
    def __init__(self):
        super().__init__()
        self._register_dsl_specific()

    def _register_dsl_specific(self):
        """
        Registers geometry-specific primitives and functions for the deterministic DSL.
        
        Sets up the vocabulary for geometric operations including basic shapes,
        transformation functions, and mathematical operations beyond those provided
        by the base class.
        """
        self.primitives.update({
            "l": _line, "c": _circle, "r": _rectangle,
        })
        self.implementations.update({
            "M": _make_affine_matrix, "T": _apply_transform, "C": lambda s1, s2: s1 + s2,
            "repeat": _repeat, "tan": math.tan, "cos": math.cos, "sin": math.sin,
        })

    def evaluate(self, node: AstNode | float | int):
        """
        Evaluates an AST node to produce geometric strokes (stateless evaluation).
        
        Recursively processes the AST node tree, resolving primitives to their
        geometric representations and applying functions to transform or combine
        strokes. This is a pure function with no side effects.
        
        Args:
            node: AstNode to evaluate, or atomic value (float/int)
            
        Returns:
            List of stroke arrays for geometric primitives/functions, or
            the atomic value itself for numbers
            
        Raises:
            ValueError: If an unknown function or primitive is encountered
            
        Examples:
            >>> renderer = DeterministicRenderer()
            >>> renderer.evaluate(AstNode('l', []))  # Basic line
            >>> renderer.evaluate(AstNode('T', [AstNode('l', []), 2.0]))  # Transformed line
        """
        if not isinstance(node, AstNode):
            return node
        if node.name in self.primitives:
            return self.primitives[node.name]
        evaluated_args = [self.evaluate(arg) for arg in node.args]
        if node.name in self.implementations:
            func = self.implementations[node.name]
            return func(*evaluated_args)
        raise ValueError(f"Unknown function or primitive: {node.name}")