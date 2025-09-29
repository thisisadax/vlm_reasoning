import math
import numpy as np
import random
from math import cos, sin, pi
from random import uniform
from ..core import AstNode, XYLIM
from .base import BaseRenderer

# --- Constants ---
RANDOM_X_POS = (-3.0, 3.0)
RANDOM_Y_POS = (-3.0, 3.0)
RANDOM_LENGTH = (1.0, 3.0)
RANDOM_ANGLE = (0, 2 * pi)
RANDOM_RADIUS = (0.5, 1.5)
RANDOM_TURN_ANGLE = (math.radians(90), math.radians(120))
RANDOM_VAR = "?"

# --- Custom Exception ---
class GeometryConstraintError(Exception):
    """Raised when a generated shape violates a geometric rule during resolution."""
    pass

# --- Scene Graph Data Structure ---
class SceneNode:
    """Represents a fully resolved object, storing both strokes and sampling functions."""
    # Add a unique ID for better debugging
    _id_counter = 0
    def __init__(self, primitive_type, children=None, constraint=None, **params):
        self.id = SceneNode._id_counter
        SceneNode._id_counter += 1
        self.primitive_type = primitive_type
        self.params = params
        self.children = children or []
        self.constraint = constraint
        self.strokes = []
        self.attachment_samplers = []
        self.world_angle = 0

class RelationalRenderer(BaseRenderer):
    """
    Renders a relational DSL using function-based attachment points for dynamic joins.
    """
    def __init__(self, max_retries=10_000, bounds=XYLIM, min_line_len=0.1):
        super().__init__()
        self.primitives = {'line', 'circle'}
        self.max_retries = max_retries
        self.bounds = bounds
        self.min_line_len = min_line_len

    def evaluate(self, node: AstNode):
        """Public entry point: Manages the retry loop for the unified resolver."""
        for _ in range(self.max_retries):
            try:
                # Reset the ID counter for each new attempt to keep logs clean
                SceneNode._id_counter = 0
                resolved_scene = self._resolve_geometry_recursive(node)
                return resolved_scene.strokes
            except GeometryConstraintError:
                continue
        raise RuntimeError(f"Failed to generate valid geometry after {self.max_retries} attempts.")

    def _resolve_geometry_recursive(self, node: AstNode) -> SceneNode:
        """A single traversal that builds and resolves the scene graph."""
        if node.name in self.primitives:
            params = self._resolve_placeholder_args(node.name, node.args)
            scene_node = SceneNode(primitive_type=node.name, **params)
            
            if node.name == 'line':
                self._draw_line(scene_node)
            elif node.name == 'circle':
                self._draw_circle(scene_node)
            
            self._validate_bounds(scene_node)
            return scene_node

        if node.name == 'join':
            if len(node.args) != 4:
                raise ValueError("The 'join' function requires 4 arguments: shape1, shape2, index1, index2.")
            
            resolved_child1 = self._resolve_geometry_recursive(node.args[0])
            resolved_child2 = self._resolve_geometry_recursive(node.args[1])
            constraint = (int(node.args[2]), int(node.args[3]))
            
            join_node = SceneNode(
                primitive_type='join',
                children=[resolved_child1, resolved_child2],
                constraint=constraint
            )
            
            self._resolve_join(join_node)
            self._validate_bounds(join_node)
            return join_node

        raise ValueError(f"Unknown function or primitive: {node.name}")

    def _resolve_placeholder_args(self, name: str, args: list) -> dict:
        """Replaces every '?' with a new random sample."""
        def sample(arg, default_range):
            is_random = isinstance(arg, AstNode) and arg.name == RANDOM_VAR
            if not is_random and isinstance(arg, AstNode):
                 raise TypeError(f"float() argument must be a string or a real number, not 'AstNode' like '{arg.name}'")
            return uniform(*default_range) if is_random else float(arg)
        
        if name == 'line':
            return {'x': sample(args[0], RANDOM_X_POS), 'y': sample(args[1], RANDOM_Y_POS),
                    'length': sample(args[2], RANDOM_LENGTH), 'angle': sample(args[3], RANDOM_ANGLE)}
        elif name == 'circle':
            return {'x': sample(args[0], RANDOM_X_POS), 'y': sample(args[1], RANDOM_Y_POS),
                    'radius': sample(args[2], RANDOM_RADIUS)}
        return {}

    def _draw_line(self, node: SceneNode):
        """Calculates geometry and creates static samplers for a line's endpoints."""
        length, angle = node.params['length'], node.params['angle']
        if length < self.min_line_len:
            raise GeometryConstraintError(f"Line length {length:.2f} is below minimum.")
        start_pos = np.array([node.params['x'], node.params['y']])
        end_pos = start_pos + np.array([length * cos(angle), length * sin(angle)])
        node.strokes = [np.array([start_pos, end_pos])]
        node.attachment_samplers = [lambda: start_pos, lambda: end_pos]
        node.world_angle = angle

    def _draw_circle(self, node: SceneNode):
        """Calculates geometry and creates a sampler that picks a random vertex."""
        radius = node.params['radius']
        center = np.array([node.params['x'], node.params['y']])
        
        # The stroke is a discrete set of points
        points = np.array([center + np.array([radius * cos(t), radius * sin(t)])
                           for t in np.linspace(0, 2 * pi, 60)])
        node.strokes = [points]
        
        # The sampler for the perimeter now randomly chooses from the actual vertices
        def create_perimeter_sampler(verts):
            return lambda: random.choice(verts)

        node.attachment_samplers = [
            lambda c=center: c, 
            create_perimeter_sampler(points)
        ]

    def _resolve_join(self, node: SceneNode):
        """Joins shapes by applying correct, pivot-based transformations."""
        shape1, shape2 = node.children
        s1_index, s2_index = node.constraint
        
        # 1. Sample the world-space points to connect
        anchor_point = shape1.attachment_samplers[s1_index]()
        pivot_point = shape2.attachment_samplers[s2_index]()
        
        # 2. Determine the correct reference angle for the join
        s1_angle = shape1.world_angle
        if shape1.primitive_type == 'circle' and s1_index == 1:
            center_point = shape1.attachment_samplers[0]()
            s1_angle = math.atan2(anchor_point[1] - center_point[1], anchor_point[0] - center_point[0])

        # 3. Calculate the required rotation for shape2
        turn_angle = uniform(*RANDOM_TURN_ANGLE) * (1 if uniform(0,1) < 0.5 else -1)
        target_angle = s1_angle + turn_angle
        rotation_to_apply = target_angle - shape2.world_angle
        
        cos_r, sin_r = cos(rotation_to_apply), sin(rotation_to_apply)
        rot_mat = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        
        # 4. **FIXED**: Apply a pivot-based transformation
        shape2.strokes = [np.dot(s - pivot_point, rot_mat.T) + anchor_point for s in shape2.strokes]
        
        transformed_samplers_s2 = [
            (lambda s=s, p=pivot_point, r=rot_mat, a=anchor_point: np.dot(s() - p, r.T) + a)
            for s in shape2.attachment_samplers
        ]
        
        # 5. Update the composite node's state
        shape2.world_angle += rotation_to_apply
        node.world_angle = shape2.world_angle
        
        # 6. **FIXED**: Intelligently merge samplers to prevent duplicates
        new_samplers = list(shape1.attachment_samplers)
        for i, sampler in enumerate(transformed_samplers_s2):
            if i != s2_index:
                new_samplers.append(sampler)
        
        node.strokes = shape1.strokes + shape2.strokes
        node.attachment_samplers = new_samplers


    def _validate_bounds(self, node: SceneNode):
        """Raises an error if any stroke geometry is outside the canvas."""
        if not node.strokes: 
            return
        if np.any(np.abs(np.vstack(node.strokes)) > self.bounds):
            raise GeometryConstraintError("Strokes exceeded canvas bounds.")