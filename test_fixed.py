#!/usr/bin/env python3
"""Test fixed polygon generation."""
import sys
sys.path.append('/scratch/gpfs/nb0564/vlm_reasoning')

from tasks.regular_polygons_variants import _generate_closed_polygon, _build_program
from renderer.languages.polygonal import PolygonalRenderer
from renderer.core import parse_program, render_strokes_to_image, export_image
import os

# Test configurations
test_configs = [
    {"shape_type": "triangle", "scale": 1.2, "rotation": 0},
    {"shape_type": "square", "scale": 1.0, "rotation": 45},
    {"shape_type": "pentagon", "scale": 1.4, "rotation": 90},
    {"shape_type": "hexagon", "scale": 0.8, "rotation": 0, "transform_type": "radial", "n_copies": 4},
]

renderer = PolygonalRenderer()
os.makedirs("test_fixed", exist_ok=True)

print("Testing fixed polygon generation...")
for i, config in enumerate(test_configs):
    print(f"\nTest {i}: {config}")
    program = _build_program(config)
    
    # Check it renders
    ast = parse_program(program)
    strokes = renderer.evaluate(ast)
    img = render_strokes_to_image(strokes, canvas_dim=512, coord_bound=5.0, line_width=3.0)
    out_path = f"test_fixed/test_{i}.png"
    export_image(img, out_path)
    print(f"  Saved to {out_path}")

# Test basic triangle to verify closure
print("\n\nTesting polygon closure...")
triangle = _generate_closed_polygon(3, 2.0)
print(f"Triangle program: {triangle[:100]}...")
ast = parse_program(triangle)
strokes = renderer.evaluate(ast)
print(f"Number of strokes: {len(strokes)}")
print("✅ Test complete! Check test_fixed/")

