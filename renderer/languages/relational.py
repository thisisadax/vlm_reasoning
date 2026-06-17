import math
import numpy as np
import random
from math import cos, sin, pi
from random import uniform
from ..core import AstNode, XYLIM
from .base import BaseRenderer

RANDOM_X_POS = (-3.0, 3.0)
RANDOM_Y_POS = (-3.0, 3.0)
RANDOM_LENGTH = (1.0, 3.0)
RANDOM_ANGLE = (0, 2 * pi)
RANDOM_RADIUS = (0.5, 1.5)
RANDOM_TURN_ANGLE = (math.radians(90), math.radians(120))
RANDOM_VAR = "?"

class GeometryConstraintError(Exception):
    """Raised when generated shape violates geometric constraints."""
    pass

class SceneNode:
    """Fully resolved object storing strokes and attachment samplers."""
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
    Supports primitives: line, circle, triangle, quadrilateral, and polygon.
    """
    def __init__(self, max_retries=10_000, bounds=XYLIM, min_line_len=0.1):
        super().__init__()
        self.primitives = {'line', 'circle', 'triangle', 'quadrilateral', 'polygon'}
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
            elif node.name == 'triangle':
                self._draw_triangle(scene_node)
            elif node.name == 'quadrilateral':
                self._draw_quadrilateral(scene_node)
            elif node.name == 'polygon':
                self._draw_polygon(scene_node)

            self._validate_bounds(scene_node)
            return scene_node

        if node.name == 'join':
            # Determine number of shapes by checking arg types
            # Shapes are AstNode objects (primitives or nested joins)
            num_shapes = 0
            for arg in node.args:
                if isinstance(arg, AstNode) and (arg.name in self.primitives or arg.name == 'join'):
                    num_shapes += 1
                else:
                    break  # First non-shape arg indicates end of shapes

            if num_shapes < 2:
                raise ValueError("Join requires at least 2 shapes")

            # Resolve all shapes recursively
            resolved_shapes = [self._resolve_geometry_recursive(node.args[i]) for i in range(num_shapes)]

            # Parse indices and optional join angles
            # For N shapes, we need (N-1) pairs of indices
            remaining_args = node.args[num_shapes:]

            # Expected args: (N-1)*2 indices, optionally followed by (N-1) angles
            min_indices = (num_shapes - 1) * 2

            # Handle 2-shape join with backward compatibility (4 or 5 args)
            if num_shapes == 2:
                if len(node.args) not in [4, 5]:
                    raise ValueError("2-shape join requires 4 or 5 arguments: shape1, shape2, index1, index2, [join_angle].")

                s1_index = int(node.args[2])
                s2_index = int(node.args[3])

                # Optional join angle (5th argument)
                join_angle = None
                if len(node.args) == 5:
                    arg = node.args[4]
                    if isinstance(arg, AstNode) and arg.name == RANDOM_VAR:
                        join_angle = uniform(*RANDOM_TURN_ANGLE) * (1 if uniform(0,1) < 0.5 else -1)
                    else:
                        join_angle = float(arg)

                # Store as list for consistency with N-shape case
                join_specs = [(s1_index, s2_index, join_angle)]
            else:
                # N-shape join (N > 2)
                if len(remaining_args) < min_indices:
                    raise ValueError(
                        f"Join with {num_shapes} shapes requires at least {min_indices} indices "
                        f"({num_shapes-1} pairs), got {len(remaining_args)}"
                    )

                # Parse index pairs (no angle support for N>2 in initial implementation)
                join_specs = []
                for i in range(0, min_indices, 2):
                    idx1 = int(remaining_args[i])
                    idx2 = int(remaining_args[i + 1])
                    join_specs.append((idx1, idx2, None))

            join_node = SceneNode(
                primitive_type='join',
                children=resolved_shapes,
                constraint=join_specs  # Now a list of (idx1, idx2, angle) tuples
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
            # Single line mode: 4 args (x, y, length, angle)
            # Multi-line mode: 6 args (x, y, num_lines, length, start_angle, angle_between)
            if len(args) == 4:
                return {'x': sample(args[0], RANDOM_X_POS),
                        'y': sample(args[1], RANDOM_Y_POS),
                        'length': sample(args[2], RANDOM_LENGTH),
                        'angle': sample(args[3], RANDOM_ANGLE),
                        'num_lines': 1}
            elif len(args) == 6:
                return {'x': sample(args[0], RANDOM_X_POS),
                        'y': sample(args[1], RANDOM_Y_POS),
                        'num_lines': int(sample(args[2], (1, 6))),
                        'length': sample(args[3], RANDOM_LENGTH),
                        'angle': sample(args[4], RANDOM_ANGLE),  # start_angle
                        'angle_between': sample(args[5], (math.radians(15), math.radians(90)))}
            else:
                raise ValueError(f"Line primitive requires 4 or 6 arguments, got {len(args)}")
        elif name == 'circle':
            return {'x': sample(args[0], RANDOM_X_POS), 'y': sample(args[1], RANDOM_Y_POS),
                    'radius': sample(args[2], RANDOM_RADIUS)}
        elif name == 'triangle':
            # Triangle with angle control: x, y, side_length, angle1, angle2, rotation
            return {'x': sample(args[0], RANDOM_X_POS),
                    'y': sample(args[1], RANDOM_Y_POS),
                    'side_length': sample(args[2], RANDOM_LENGTH),
                    'angle1': sample(args[3], (math.radians(30), math.radians(120))),
                    'angle2': sample(args[4], (math.radians(30), math.radians(120))),
                    'rotation': sample(args[5], RANDOM_ANGLE)}
        elif name == 'quadrilateral':
            # Quadrilateral with angle control: x, y, scale, angle1, angle2, angle3, rotation
            # angle4 will be computed as 360° - (angle1 + angle2 + angle3)
            return {'x': sample(args[0], RANDOM_X_POS),
                    'y': sample(args[1], RANDOM_Y_POS),
                    'scale': sample(args[2], RANDOM_LENGTH),
                    'angle1': sample(args[3], (math.radians(60), math.radians(120))),
                    'angle2': sample(args[4], (math.radians(60), math.radians(120))),
                    'angle3': sample(args[5], (math.radians(60), math.radians(120))),
                    'rotation': sample(args[6], RANDOM_ANGLE)}
        elif name == 'polygon':
            # N-sided polygon: x, y, n_sides, radius, rotation
            # n_sides should be >= 5 (triangles and quads have their own primitives)
            return {'x': sample(args[0], RANDOM_X_POS),
                    'y': sample(args[1], RANDOM_Y_POS),
                    'n_sides': int(sample(args[2], (5, 12))),  # Default random: 5-12 sides
                    'radius': sample(args[3], RANDOM_RADIUS),
                    'rotation': sample(args[4], RANDOM_ANGLE)}
        return {}

    def _draw_line(self, node: SceneNode):
        """
        Calculates geometry for single or multiple lines emanating from a point.

        Single line mode: Creates one line from (x, y) at specified angle
        Multi-line mode: Creates multiple lines from same origin with specified angles between them
        """
        x, y = node.params['x'], node.params['y']
        length = node.params['length']
        num_lines = node.params.get('num_lines', 1)

        if length < self.min_line_len:
            raise GeometryConstraintError(f"Line length {length:.2f} is below minimum.")

        origin = np.array([x, y])
        strokes = []
        attachment_points = [origin]  # Origin is always an attachment point

        if num_lines == 1:
            # Single line
            angle = node.params['angle']
            end_pos = origin + np.array([length * cos(angle), length * sin(angle)])
            strokes.append(np.array([origin, end_pos]))
            attachment_points.append(end_pos)
            node.world_angle = angle
        else:
            # Multi-line mode
            start_angle = node.params['angle']
            angle_between = node.params.get('angle_between', 0)

            # Validate num_lines
            if num_lines < 1 or num_lines > 6:
                raise GeometryConstraintError(f"Number of lines must be 1-6, got {num_lines}")

            # Create lines radiating from origin
            for i in range(num_lines):
                current_angle = start_angle + i * angle_between
                end_pos = origin + np.array([length * cos(current_angle), length * sin(current_angle)])
                strokes.append(np.array([origin, end_pos]))
                attachment_points.append(end_pos)

            # World angle is the average direction (for join operations)
            node.world_angle = start_angle + (num_lines - 1) * angle_between / 2

        node.strokes = strokes

        # Create attachment samplers: origin + all endpoints
        node.attachment_samplers = [
            lambda p=pt.copy(): p for pt in attachment_points
        ]

    def _draw_circle(self, node: SceneNode):
        """Calculates geometry and creates a sampler that picks a random vertex."""
        radius = node.params['radius']
        center = np.array([node.params['x'], node.params['y']])
        
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
    
    def _draw_triangle(self, node: SceneNode):
        """
        Draws a triangle with controllable internal angles.

        Uses angle1 and angle2 to define the triangle shape. The third angle
        is automatically computed to ensure angles sum to 180°.
        Uses law of sines to compute side lengths, then constructs vertices.
        """
        x, y = node.params['x'], node.params['y']
        base_length = node.params['side_length']
        angle1 = node.params['angle1']  # Angle at vertex 0
        angle2 = node.params['angle2']  # Angle at vertex 1
        rot = node.params['rotation']

        # Validate angles
        if angle1 + angle2 >= math.pi:
            raise GeometryConstraintError(
                f"Triangle angles invalid: angle1={math.degrees(angle1):.1f}°, "
                f"angle2={math.degrees(angle2):.1f}° (sum must be < 180°)"
            )

        angle3 = math.pi - angle1 - angle2  # Angle at vertex 2

        # Construct triangle using base_length as the side opposite to angle3
        # Place v0 at origin, v1 along x-axis
        v0 = np.array([0.0, 0.0])
        v1 = np.array([base_length, 0.0])

        # Use law of sines to find other side lengths
        # side0 (opposite angle1) / sin(angle1) = base_length / sin(angle3)
        # side1 (opposite angle2) / sin(angle2) = base_length / sin(angle3)
        if math.sin(angle3) < 1e-6:
            raise GeometryConstraintError(f"Degenerate triangle: angle3={math.degrees(angle3):.1f}° too small")

        side1 = base_length * math.sin(angle2) / math.sin(angle3)  # Length from v0 to v2

        # Position v2 using angle1 at v0
        v2_x = side1 * math.cos(angle1)
        v2_y = side1 * math.sin(angle1)
        v2 = np.array([v2_x, v2_y])

        vertices = np.array([v0, v1, v2])

        # Center the triangle at its centroid
        centroid = vertices.mean(axis=0)
        vertices -= centroid

        # Apply rotation
        cos_r, sin_r = math.cos(rot), math.sin(rot)
        rot_mat = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        rotated = vertices @ rot_mat.T

        # Apply translation
        translated = rotated + np.array([x, y])

        # Create strokes (closed path)
        node.strokes = [np.vstack([translated, translated[0:1]])]

        # Create attachment samplers for each vertex
        node.attachment_samplers = [
            lambda v=translated[i].copy(): v for i in range(3)
        ]

        # Store world angle for join operations
        node.world_angle = rot

    def _draw_rectangle(self, node: SceneNode):
        """Draws a rectangle with attachment points at vertices."""
        x, y = node.params['x'], node.params['y']
        width = node.params['width']
        height = node.params['height']
        rot = node.params['rotation']
        
        # can specify with just bottom left, upper right but for precision use 4
        vertices = np.array([
            [-width/2, -height/2],    # Bottom-left
            [width/2, -height/2],     # Bottom-right
            [width/2, height/2],      # Top-right
            [-width/2, height/2],     # Top-left
        ])
        
        # Apply rotation
        cos_r, sin_r = math.cos(rot), math.sin(rot)
        rot_mat = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        rotated = vertices @ rot_mat.T
        
        # Apply translation
        translated = rotated + np.array([x, y])
        
        # Create strokes (closed path)
        node.strokes = [np.vstack([translated, translated[0:1]])]
        
        # Create attachment samplers for each vertex
        node.attachment_samplers = [
            lambda v=translated[i].copy(): v for i in range(4)
        ]
        
        # Store world angle for join operations
        node.world_angle = rot
        node.center = np.array([x, y])
        node.width = width
        node.height = height
        #finish adding rectangle

    def _draw_quadrilateral(self, node: SceneNode):
        """
        Draws a quadrilateral with controllable internal angles.

        Uses angle1, angle2, angle3 to define the quadrilateral shape.
        The fourth angle is computed to ensure angles sum to 360°.
        Constructs vertices sequentially using the angles.
        """
        x, y = node.params['x'], node.params['y']
        scale = node.params['scale']
        angle1 = node.params['angle1']  # Internal angle at vertex 0
        angle2 = node.params['angle2']  # Internal angle at vertex 1
        angle3 = node.params['angle3']  # Internal angle at vertex 2
        rot = node.params['rotation']

        # Compute the fourth angle
        angle4 = 2 * math.pi - (angle1 + angle2 + angle3)

        # Validate angles
        if angle4 <= 0 or angle4 >= math.pi:
            raise GeometryConstraintError(
                f"Quadrilateral angles invalid: angle4={math.degrees(angle4):.1f}° "
                f"(computed from angle1={math.degrees(angle1):.1f}°, "
                f"angle2={math.degrees(angle2):.1f}°, angle3={math.degrees(angle3):.1f}°)"
            )

        # Construct quadrilateral by sequential vertex placement
        # Start with v0 at origin, v1 along positive x-axis
        v0 = np.array([0.0, 0.0])
        v1 = np.array([scale, 0.0])

        # From v1, we turn by the exterior angle (180° - angle1) and place v2
        # Direction from v0 to v1 is 0 radians
        # At v1, the interior angle is angle1
        # Exterior angle = 180° - angle1
        # New direction = 0 + (180° - angle1) = π - angle1
        dir_v1_to_v2 = math.pi - angle1
        v2 = v1 + scale * np.array([math.cos(dir_v1_to_v2), math.sin(dir_v1_to_v2)])

        # From v2, turn by exterior angle (180° - angle2)
        # Current direction is dir_v1_to_v2
        # New direction after turning by exterior angle at v2
        dir_v2_to_v3 = dir_v1_to_v2 + (math.pi - angle2)
        v3 = v2 + scale * np.array([math.cos(dir_v2_to_v3), math.sin(dir_v2_to_v3)])

        # The 4th vertex is already determined - it's v0, but let's verify closure
        # The vector from v3 to v0 should align with our angle constraints
        # We don't place v4 separately; the quadrilateral closes at v0

        vertices = np.array([v0, v1, v2, v3])

        # Center the quadrilateral at its centroid
        centroid = vertices.mean(axis=0)
        vertices -= centroid

        # Apply rotation
        cos_r, sin_r = math.cos(rot), math.sin(rot)
        rot_mat = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        rotated = vertices @ rot_mat.T

        # Apply translation
        translated = rotated + np.array([x, y])

        # Create strokes (closed path)
        node.strokes = [np.vstack([translated, translated[0:1]])]

        # Create attachment samplers for each vertex
        node.attachment_samplers = [
            lambda v=translated[i].copy(): v for i in range(4)
        ]

        # Store world angle for join operations
        node.world_angle = rot

    def _draw_polygon(self, node: SceneNode):
        """
        Draws a regular n-sided polygon with controllable number of sides.

        For n >= 50, this approximates a circle.
        All vertices are equidistant from center (regular polygon).
        """
        x, y = node.params['x'], node.params['y']
        n_sides = node.params['n_sides']
        radius = node.params['radius']
        rot = node.params['rotation']

        # Validate n_sides
        if n_sides < 3:
            raise GeometryConstraintError(f"Polygon must have at least 3 sides, got {n_sides}")

        # For very high n, treat as circle for smoothness
        if n_sides >= 50:
            n_points = 60  # Use 60 points for smooth circle rendering
        else:
            n_points = n_sides

        # Generate vertices around a circle
        angles = np.linspace(0, 2 * math.pi, n_points, endpoint=False)
        vertices = np.column_stack([
            radius * np.cos(angles),
            radius * np.sin(angles)
        ])

        # Apply rotation
        cos_r, sin_r = math.cos(rot), math.sin(rot)
        rot_mat = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        rotated = vertices @ rot_mat.T

        # Apply translation
        translated = rotated + np.array([x, y])

        # Create strokes (closed path)
        node.strokes = [np.vstack([translated, translated[0:1]])]

        # Create attachment samplers for each vertex
        # For regular polygons, provide samplers for the actual n_sides vertices
        if n_sides < 50:
            # Use actual polygon vertices as attachment points
            node.attachment_samplers = [
                lambda v=translated[i].copy(): v for i in range(n_sides)
            ]
        else:
            # For circles, provide center + random perimeter sampler
            center = np.array([x, y])
            def create_perimeter_sampler(verts):
                return lambda: random.choice(verts)
            node.attachment_samplers = [
                lambda c=center: c.copy(),
                create_perimeter_sampler(translated)
            ]

        # Store world angle for join operations
        node.world_angle = rot


    def _draw_ellipse(self, node: SceneNode):
        """Draws an ellipse with center and perimeter attachment points."""
        x, y = node.params['x'], node.params['y']
        w, h = node.params['width'], node.params['height']
        rot = node.params['rotation']
        
    def _resolve_join(self, node: SceneNode):
        """Joins N shapes sequentially using pivot-based transformations."""
        shapes = node.children
        join_specs = node.constraint

        if len(shapes) != len(join_specs) + 1:
            raise ValueError(
                f"Inconsistent join: {len(shapes)} shapes requires {len(shapes)-1} join specs, "
                f"got {len(join_specs)}"
            )

        #initialize composite with first shape
        composite_strokes = list(shapes[0].strokes)
        composite_samplers = list(shapes[0].attachment_samplers)
        composite_world_angle = shapes[0].world_angle
        composite_primitive_type = shapes[0].primitive_type

        #iteratively join each subsequent shape to the growing composite
        for i, next_shape in enumerate(shapes[1:]):
            idx1, idx2, join_angle = join_specs[i]

            if idx1 >= len(composite_samplers):
                raise ValueError(
                    f"Join {i+1}: composite attachment index {idx1} out of range "
                    f"(composite has {len(composite_samplers)} attachment points)"
                )
            if idx2 >= len(next_shape.attachment_samplers):
                raise ValueError(
                    f"Join {i+1}: shape attachment index {idx2} out of range "
                    f"(shape has {len(next_shape.attachment_samplers)} attachment points)"
                )

            #sample world-space connection points
            anchor_point = composite_samplers[idx1]()
            pivot_point = next_shape.attachment_samplers[idx2]()

            #determine reference angle for join
            anchor_angle = composite_world_angle
            if composite_primitive_type == 'circle' and idx1 == 1:
                center_point = composite_samplers[0]()
                anchor_angle = math.atan2(
                    anchor_point[1] - center_point[1],
                    anchor_point[0] - center_point[0]
                )

            #calculate rotation for next_shape
            if join_angle is not None:
                turn_angle = join_angle
            else:
                turn_angle = uniform(*RANDOM_TURN_ANGLE) * (1 if uniform(0,1) < 0.5 else -1)

            target_angle = anchor_angle + turn_angle
            rotation_to_apply = target_angle - next_shape.world_angle

            cos_r, sin_r = cos(rotation_to_apply), sin(rotation_to_apply)
            rot_mat = np.array([[cos_r, -sin_r], [sin_r, cos_r]])

            #apply pivot-based transformation to next_shape
            transformed_strokes = [
                np.dot(s - pivot_point, rot_mat.T) + anchor_point
                for s in next_shape.strokes
            ]

            transformed_samplers = [
                (lambda s=s, p=pivot_point, r=rot_mat, a=anchor_point:
                 np.dot(s() - p, r.T) + a)
                for s in next_shape.attachment_samplers
            ]

            #update composite by merging strokes and attachment points
            composite_strokes.extend(transformed_strokes)
            for j, sampler in enumerate(transformed_samplers):
                if j != idx2:
                    composite_samplers.append(sampler)

            composite_world_angle = next_shape.world_angle + rotation_to_apply
            composite_primitive_type = 'join'

        node.strokes = composite_strokes
        node.attachment_samplers = composite_samplers
        node.world_angle = composite_world_angle


    def _validate_bounds(self, node: SceneNode):
        """Raises an error if any stroke geometry is outside the canvas."""
        if not node.strokes: 
            return
        if np.any(np.abs(np.vstack(node.strokes)) > self.bounds):
            raise GeometryConstraintError("Strokes exceeded canvas bounds.")