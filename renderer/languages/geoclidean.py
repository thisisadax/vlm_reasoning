"""
Geoclidean DSL Renderer for the unified 47-concept dataset.

This module parses and renders Geoclidean concept definitions using constraint-based
geometric construction. It outputs strokes compatible with the Cairo renderer in core.py.

Geoclidean DSL Syntax:
    'l1 = line(p1(), p2())'           # Line from random point p1 to random point p2
    'c1 = circle(p1(), p2())'         # Circle with center p1, radius = dist(p1, p2)
    'l1* = line(...)'                 # Construction element (invisible in final render)
    'p3(c1)'                          # Random point on circle c1
    'p3(l1)'                          # Random point on line l1
    'p3(c1, c2)'                      # Intersection of circle c1 and circle c2
    'p3(l1, l2)'                      # Intersection of line l1 and line l2
    'p3(l1, c1)'                      # Intersection of line l1 and circle c1

"""

import re
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from ..core import XYLIM

# --- Constants ---
RANDOM_X_POS = (-3.0, 3.0)
RANDOM_Y_POS = (-3.0, 3.0)
CIRCLE_POINTS = 60  # Number of points to discretize circle
MIN_LINE_LENGTH = 0.3
MIN_CIRCLE_RADIUS = 0.3
MAX_CIRCLE_RADIUS = 3.5


class GeometryConstraintError(Exception):
    """Raised when geometry generation fails constraints."""
    pass


class GeoObject:
    """Base class for geometric objects (lines and circles)."""
    def __init__(self, name: str, visible: bool = True):
        self.name = name
        self.visible = visible
        self.strokes = []


class GeoLine(GeoObject):
    """A line segment defined by two endpoints."""
    def __init__(self, name: str, p1: np.ndarray, p2: np.ndarray, visible: bool = True):
        super().__init__(name, visible)
        self.p1 = np.array(p1, dtype=np.float64)
        self.p2 = np.array(p2, dtype=np.float64)
        self.strokes = [np.array([self.p1, self.p2])]

    def point_on_line(self, t: float) -> np.ndarray:
        """Get point at parameter t in [0, 1] along the line."""
        return self.p1 + t * (self.p2 - self.p1)

    def random_point(self) -> np.ndarray:
        """Get a random point on this line segment."""
        t = np.random.uniform(0.1, 0.9)  # Avoid exact endpoints
        return self.point_on_line(t)

    def length(self) -> float:
        """Return line length."""
        return np.linalg.norm(self.p2 - self.p1)


class GeoCircle(GeoObject):
    """A circle defined by center and radius."""
    def __init__(self, name: str, center: np.ndarray, radius: float, visible: bool = True):
        super().__init__(name, visible)
        self.center = np.array(center, dtype=np.float64)
        self.radius = float(radius)
        # Generate discretized circle stroke
        theta = np.linspace(0, 2 * np.pi, CIRCLE_POINTS)
        points = np.column_stack([
            self.center[0] + self.radius * np.cos(theta),
            self.center[1] + self.radius * np.sin(theta)
        ])
        self.strokes = [points]

    def random_point(self) -> np.ndarray:
        """Get a random point on this circle's perimeter."""
        theta = np.random.uniform(0, 2 * np.pi)
        return self.center + self.radius * np.array([np.cos(theta), np.sin(theta)])


def line_line_intersection(l1: GeoLine, l2: GeoLine) -> Optional[np.ndarray]:
    """
    Compute intersection of two lines (extended to infinite lines).
    Returns None if parallel.
    """
    x1, y1 = l1.p1
    x2, y2 = l1.p2
    x3, y3 = l2.p1
    x4, y4 = l2.p2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None  # Parallel lines

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

    px = x1 + t * (x2 - x1)
    py = y1 + t * (y2 - y1)
    return np.array([px, py])


def line_circle_intersection(line: GeoLine, circle: GeoCircle) -> List[np.ndarray]:
    """
    Compute intersection(s) of a line (extended) with a circle.
    Returns list of 0, 1, or 2 intersection points.
    """
    # Line direction
    d = line.p2 - line.p1
    f = line.p1 - circle.center

    a = np.dot(d, d)
    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - circle.radius ** 2

    discriminant = b ** 2 - 4 * a * c

    if discriminant < -1e-10:
        return []

    points = []
    if discriminant < 1e-10:
        # Tangent
        t = -b / (2 * a)
        points.append(line.p1 + t * d)
    else:
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)
        points.append(line.p1 + t1 * d)
        points.append(line.p1 + t2 * d)

    return points


def circle_circle_intersection(c1: GeoCircle, c2: GeoCircle) -> List[np.ndarray]:
    """
    Compute intersection(s) of two circles.
    Returns list of 0, 1, or 2 intersection points.
    """
    d = np.linalg.norm(c2.center - c1.center)

    # Check for no intersection or coincident
    if d > c1.radius + c2.radius + 1e-10:
        return []
    if d < abs(c1.radius - c2.radius) - 1e-10:
        return []
    if d < 1e-10:
        return []  # Coincident centers

    a = (c1.radius ** 2 - c2.radius ** 2 + d ** 2) / (2 * d)
    h_sq = c1.radius ** 2 - a ** 2

    if h_sq < -1e-10:
        return []

    h = np.sqrt(max(0, h_sq))

    # Point on line between centers
    direction = (c2.center - c1.center) / d
    p = c1.center + a * direction

    # Perpendicular direction
    perp = np.array([-direction[1], direction[0]])

    if h < 1e-10:
        # Tangent (one point)
        return [p]
    else:
        # Two points
        return [p + h * perp, p - h * perp]


class GeoclideanRenderer:
    """
    Renderer for Geoclidean DSL concepts.

    Parses concept definition rules and generates strokes compatible
    with the Cairo renderer.
    """

    def __init__(self, max_retries: int = 5000, bounds: float = XYLIM):
        self.max_retries = max_retries
        self.bounds = bounds

    def render_concept(self, rules: List[str]) -> List[np.ndarray]:
        """
        Render a concept from its DSL rules.

        Args:
            rules: List of DSL rule strings

        Returns:
            List of stroke arrays (numpy arrays of shape (N, 2))
        """
        for _ in range(self.max_retries):
            try:
                return self._render_attempt(rules)
            except GeometryConstraintError:
                continue
        raise RuntimeError(f"Failed to generate valid geometry after {self.max_retries} attempts.")

    def _render_attempt(self, rules: List[str]) -> List[np.ndarray]:
        """Single attempt to render a concept."""
        # Storage for resolved objects and points
        objects: Dict[str, GeoObject] = {}
        points: Dict[str, np.ndarray] = {}

        for rule in rules:
            self._execute_rule(rule.strip(), objects, points)

        # Collect visible strokes
        strokes = []
        for obj in objects.values():
            if obj.visible:
                strokes.extend(obj.strokes)

        # Validate bounds
        if strokes:
            all_points = np.vstack(strokes)
            if np.any(np.abs(all_points) > self.bounds):
                raise GeometryConstraintError("Strokes exceeded canvas bounds.")

        if not strokes:
            raise GeometryConstraintError("No visible strokes generated.")

        return strokes

    def _execute_rule(self, rule: str, objects: Dict[str, GeoObject],
                      points: Dict[str, np.ndarray]) -> None:
        """
        Execute a single DSL rule.

        Examples:
            'l1 = line(p1(), p2())'
            'c1* = circle(p1(), p2())'
            'l2 = line(p3(c1, c2), p4(l1))'
        """
        # Clean up the rule string
        rule = rule.strip().strip("'").strip('"').strip(',')
        if not rule:
            return

        # Parse: name(*)? = primitive(args)
        # Handle special case in rhomboid where lines are named like '12', '13' etc.
        match = re.match(r"([a-zA-Z]?\d+)(\*)?\s*=\s*(\w+)\((.+)\)", rule)
        if not match:
            return

        obj_name = match.group(1)
        is_construction = match.group(2) == '*'
        primitive = match.group(3)
        args_str = match.group(4)

        # Parse arguments (point references)
        point_refs = self._parse_point_args(args_str)

        resolved_points = [self._resolve_point(pref, objects, points) for pref in point_refs]

        # Create the object
        visible = not is_construction
        if primitive == 'line':
            if len(resolved_points) != 2:
                raise GeometryConstraintError(f"Line requires 2 points, got {len(resolved_points)}")
            p1, p2 = resolved_points
            length = np.linalg.norm(p2 - p1)
            if length < MIN_LINE_LENGTH:
                raise GeometryConstraintError(f"Line too short: {length:.3f}")
            obj = GeoLine(obj_name, p1, p2, visible=visible)
        elif primitive == 'circle':
            if len(resolved_points) != 2:
                raise GeometryConstraintError(f"Circle requires 2 points, got {len(resolved_points)}")
            center, perimeter_pt = resolved_points
            radius = np.linalg.norm(perimeter_pt - center)
            if radius < MIN_CIRCLE_RADIUS:
                raise GeometryConstraintError(f"Circle radius too small: {radius:.3f}")
            if radius > MAX_CIRCLE_RADIUS:
                raise GeometryConstraintError(f"Circle radius too large: {radius:.3f}")
            obj = GeoCircle(obj_name, center, radius, visible=visible)
        else:
            raise ValueError(f"Unknown primitive: {primitive}")

        objects[obj_name] = obj

    def _parse_point_args(self, args_str: str) -> List[Tuple[str, List[str]]]:
        """
        Parse point argument string into list of (point_name, constraint_refs).

        Examples:
            'p1(), p2()' -> [('p1', []), ('p2', [])]
            'p3(c1), p4(c1, c2)' -> [('p3', ['c1']), ('p4', ['c1', 'c2'])]
        """
        results = []
        for match in re.finditer(r'(p\d+)\(([^)]*)\)', args_str):
            point_name = match.group(1)
            constraints_str = match.group(2).strip()
            constraints = [c.strip() for c in constraints_str.split(',')] if constraints_str else []
            results.append((point_name, constraints))
        return results

    def _resolve_point(self, point_ref: Tuple[str, List[str]],
                       objects: Dict[str, GeoObject],
                       points: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Resolve a point reference to concrete coordinates.

        Args:
            point_ref: (point_name, [constraint_object_names])
            objects: Dict of already-created objects
            points: Dict of already-resolved points

        Returns:
            numpy array of shape (2,)
        """
        point_name, constraints = point_ref

        # Check if already resolved
        if point_name in points:
            return points[point_name]

        if len(constraints) == 0:
            # Free random point
            pt = np.array([
                np.random.uniform(*RANDOM_X_POS),
                np.random.uniform(*RANDOM_Y_POS)
            ])
        elif len(constraints) == 1:
            # Point on object
            obj = objects.get(constraints[0])
            if obj is None:
                raise GeometryConstraintError(f"Unknown object: {constraints[0]}")
            if isinstance(obj, GeoCircle):
                pt = obj.random_point()
            elif isinstance(obj, GeoLine):
                pt = obj.random_point()
            else:
                raise GeometryConstraintError(f"Cannot sample point on {type(obj)}")
        elif len(constraints) == 2:
            # Intersection of two objects
            obj1 = objects.get(constraints[0])
            obj2 = objects.get(constraints[1])
            if obj1 is None or obj2 is None:
                raise GeometryConstraintError(f"Unknown objects: {constraints}")

            pt = self._compute_intersection(obj1, obj2)
        else:
            raise GeometryConstraintError(f"Too many constraints: {constraints}")

        points[point_name] = pt
        return pt

    def _compute_intersection(self, obj1: GeoObject, obj2: GeoObject) -> np.ndarray:
        """Compute intersection point between two objects."""
        if isinstance(obj1, GeoLine) and isinstance(obj2, GeoLine):
            pt = line_line_intersection(obj1, obj2)
            if pt is None:
                raise GeometryConstraintError("Lines are parallel, no intersection")
            return pt

        if isinstance(obj1, GeoLine) and isinstance(obj2, GeoCircle):
            pts = line_circle_intersection(obj1, obj2)
        elif isinstance(obj1, GeoCircle) and isinstance(obj2, GeoLine):
            pts = line_circle_intersection(obj2, obj1)
        elif isinstance(obj1, GeoCircle) and isinstance(obj2, GeoCircle):
            pts = circle_circle_intersection(obj1, obj2)
        else:
            raise GeometryConstraintError(f"Cannot intersect {type(obj1)} and {type(obj2)}")

        if len(pts) == 0:
            raise GeometryConstraintError("No intersection found")

        # Return random choice among intersection points
        return pts[np.random.randint(len(pts))]


def load_concept_rules(concept_path: Path) -> List[str]:
    """Load DSL rules from a concept.txt file."""
    with open(concept_path, 'r') as f:
        content = f.read()

    # Parse the Python-like list of strings
    rules = []
    for line in content.strip().split('\n'):
        line = line.strip().strip(',')
        if line.startswith("'") or line.startswith('"'):
            # Remove quotes
            rule = line.strip("'").strip('"')
            if rule:
                rules.append(rule)
    return rules


def load_all_geoclidean_concepts(base_path: Path) -> Dict[str, List[str]]:
    """
    Load all Geoclidean concept definitions.

    Args:
        base_path: Path to geoclidean_framework/geoclidean directory

    Returns:
        Dict mapping concept names to lists of DSL rules
    """
    concepts = {}

    # Load elements
    elements_path = base_path / 'elements'
    if elements_path.exists():
        for concept_dir in elements_path.iterdir():
            if concept_dir.is_dir() and concept_dir.name.startswith('concept_'):
                concept_file = concept_dir / 'concept.txt'
                if concept_file.exists():
                    concept_name = concept_dir.name  # e.g., 'concept_triangle'
                    concepts[concept_name] = load_concept_rules(concept_file)

    # Load constraints
    constraints_path = base_path / 'constraints'
    if constraints_path.exists():
        for concept_dir in constraints_path.iterdir():
            if concept_dir.is_dir() and concept_dir.name.startswith('concept_'):
                concept_file = concept_dir / 'concept.txt'
                if concept_file.exists():
                    concept_name = concept_dir.name
                    concepts[concept_name] = load_concept_rules(concept_file)

    return concepts


def generate_concept_strokes(concept_name: str, rules: List[str],
                             renderer: GeoclideanRenderer = None) -> List[np.ndarray]:
    """
    Generate strokes for a Geoclidean concept.

    Args:
        concept_name: Name of the concept (for error messages)
        rules: List of DSL rule strings
        renderer: GeoclideanRenderer instance (created if None)

    Returns:
        List of stroke arrays compatible with render_strokes_to_image()
    """
    if renderer is None:
        renderer = GeoclideanRenderer()
    return renderer.render_concept(rules)


# Convenience function for generating images
def render_geoclidean_concept_image(concept_name: str, rules: List[str],
                                    canvas_dim: int = 128,
                                    coord_bound: float = 5.0,
                                    line_width: float = 2.0) -> np.ndarray:
    """
    Render a Geoclidean concept to an image.

    Args:
        concept_name: Name of the concept
        rules: List of DSL rule strings
        canvas_dim: Output image size in pixels
        coord_bound: Coordinate bounds (-coord_bound to +coord_bound)
        line_width: Stroke width in pixels

    Returns:
        numpy array of shape (canvas_dim, canvas_dim, 3) with RGB values [0,1]
    """
    from ..core import render_strokes_to_image

    renderer = GeoclideanRenderer(bounds=coord_bound)
    strokes = renderer.render_concept(rules)
    return render_strokes_to_image(strokes, canvas_dim=canvas_dim,
                                   coord_bound=coord_bound, line_width=line_width)


# --- Test/Demo code ---
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from pathlib import Path

    # Test with triangle concept
    triangle_rules = [
        'l1 = line(p1(), p2())',
        'l2 = line(p2(), p3())',
        'l3 = line(p3(), p1())',
    ]

    renderer = GeoclideanRenderer()
    strokes = renderer.render_concept(triangle_rules)

    print(f"Generated {len(strokes)} strokes for triangle")
    for i, s in enumerate(strokes):
        print(f"  Stroke {i}: shape {s.shape}")

    # Test with ccc concept
    ccc_rules = [
        'c1 = circle(p1(), p2())',
        'c2 = circle(p3(c1), p4(c1))',
        'c3 = circle(p5(c1), p6(c1, c2))',
    ]

    strokes_ccc = renderer.render_concept(ccc_rules)
    print(f"\nGenerated {len(strokes_ccc)} strokes for ccc")
