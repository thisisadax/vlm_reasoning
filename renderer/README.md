# DSL Renderer

A flexible, extensible system for rendering geometric domain-specific languages (DSLs) to raster images. The renderer supports multiple DSL dialects with different rendering characteristics, from precise deterministic geometry to organic probabilistic drawing.

## Overview

This directory contains a modular DSL rendering system built around S-expression parsing and stroke-based graphics rendering. The system is designed to be easily extensible with new DSL "languages" while sharing common infrastructure for parsing, rendering, and image export.

### Key Features

- **Multiple DSL Support**: Currently supports deterministic, probabilistic, and colored geometric DSLs
- **S-Expression Parsing**: Programs written in Lisp-like syntax for easy composition
- **Stroke-Based Rendering**: Vector graphics rendered to high-quality raster images using Cairo
- **Extensible Architecture**: Clean base class system for adding new DSL variants
- **Batch Processing**: CSV-based workflow for rendering large datasets
- **CLI Interface**: Command-line tool for both single programs and batch processing

## Architecture

### Core Components

```
renderer/
├── core.py              # Parsing, rendering, and utility functions
├── render.py            # Command-line interface
└── languages/           # DSL implementations
    ├── base.py         # Abstract base renderer class
    ├── deterministic.py # Precise geometric DSL
    ├── probabilistic.py # Organic drawing DSL
    └── colored.py      # Colored variant of deterministic DSL
```

### System Flow

1. **Parse**: S-expression program strings → Abstract Syntax Tree (AST)
2. **Evaluate**: AST → List of geometric strokes (using DSL-specific renderer)
3. **Render**: Strokes → Raster image (using Cairo graphics)
4. **Export**: Image → PNG file

## Existing DSL Languages

### 1. Deterministic DSL (`languages/deterministic.py`)

A stateless, deterministic language for precise geometric constructions.

**Primitives:**
- `l` - Unit line segment from (0,0) to (1,0)
- `c` - Unit circle centered at origin with radius 0.5
- `r` - Unit rectangle centered at origin

**Functions:**
- `T strokes s theta x y` - Transform strokes with scale, rotation, translation
- `C stroke1 stroke2 ...` - Compose (concatenate) multiple stroke lists
- `repeat strokes n matrix` - Repeat strokes n times with cumulative transformation
- `M s theta x y order` - Create affine transformation matrix
- Math: `sin`, `cos`, `tan`, `+`, `-`, `*`, `/`, `pi`

**Example Programs:**
```lisp
(T l 2 1.5)                          ; Scale line by 2, rotate by 1.5 radians
(C l c)                              ; Combine line and circle
(repeat (T l 1 (/ pi 6)) 12 (M 1 (/ pi 6))) ; 12-pointed star
```

### 2. Probabilistic DSL (`languages/probabilistic.py`)

A stateful, probabilistic language that simulates natural drawing with organic variations.

**State:** Maintains current position `(x, y)` and drawing angle

**Drawing Commands:**
- `curve d_angle length bend` - Draw curved line with angle change, length, and curvature
- `dot radius` - Draw circular dot at current position
- `turn angle` - Change drawing direction without moving

**State Commands:**
- `lift x y [angle]` - Move to position without drawing
- `C cmd1 cmd2 ...` - Execute commands in sequence

**Noise Parameters:** All drawing commands include probabilistic variations for organic appearance.

**Example Programs:**
```lisp
(curve 0.5 2.0 0.3)                  ; Curved stroke
(C (curve 0 1 0) (turn 1.57) (curve 0 1 0)) ; L-shaped path
(lift 1 1 0)                         ; Move to (1,1) facing up
```

### 3. Colored DSL (`languages/colored.py`)

Extends the deterministic DSL with color support.

**Additional Function:**
- `color r g b strokes` - Apply RGB color (0-1 range) to stroke list

**Stroke Format:** Returns `(stroke_array, (r, g, b))` tuples instead of plain arrays

**Example Programs:**
```lisp
(color 1 0 0 l)                      ; Red line
(C (color 0 1 0 l) (color 0 0 1 c))  ; Green line + blue circle
```

## Adding New DSL Languages

To create a new DSL language, follow these steps:

### 1. Create Language File

Create a new file in `languages/your_language.py`:

```python
from .base import BaseRenderer
from ..core import AstNode
import numpy as np

class YourRenderer(BaseRenderer):
    def __init__(self):
        super().__init__()
        self._register_dsl_specific()
    
    def _register_dsl_specific(self):
        """Register your DSL's primitives and functions"""
        # Add primitives (constants, basic shapes)
        self.primitives.update({
            "myshape": [np.array([[0, 0], [1, 0], [0.5, 1], [0, 0]])],
        })
        
        # Add function implementations  
        self.implementations.update({
            "myfunction": self._my_function,
        })
    
    def _my_function(self, arg1, arg2):
        """Example function implementation"""
        # Your function logic here
        return result_strokes
    
    def evaluate(self, node: AstNode):
        """Main evaluation method - customize based on your DSL semantics"""
        # For stateless DSLs, use pattern similar to DeterministicRenderer
        # For stateful DSLs, use pattern similar to ProbabilisticRenderer
        pass
```

### 2. Required Methods

Your renderer class must implement:

**`__init__()`**: Initialize and call `_register_dsl_specific()`

**`_register_dsl_specific()`**: Register your DSL's vocabulary:
- `self.primitives[name] = value` - Constants and basic shapes
- `self.implementations[name] = function` - Function implementations

**`evaluate(node: AstNode)`**: Main evaluation method that returns stroke list

### 3. Stroke Format

Your `evaluate()` method must return a list where each element is either:
- **Basic stroke**: `numpy.ndarray` of shape `(N, 2)` with 2D points
- **Colored stroke**: `(stroke_array, (r, g, b))` tuple

### 4. Common Patterns

**Stateless DSL** (like deterministic):
```python
def evaluate(self, node):
    if not isinstance(node, AstNode):
        return node  # Return atomic values as-is
    if node.name in self.primitives:
        return self.primitives[node.name]  # Return primitive value
    # Evaluate arguments recursively
    args = [self.evaluate(arg) for arg in node.args]
    if node.name in self.implementations:
        return self.implementations[node.name](*args)
    raise ValueError(f"Unknown: {node.name}")
```

**Stateful DSL** (like probabilistic):
```python
def evaluate(self, node):
    initial_state = {'pos': np.array([0., 0.]), 'angle': 0.}
    strokes, _ = self._evaluate_recursive(node, initial_state)
    return strokes

def _evaluate_recursive(self, node, state):
    # Handle composition and state-modifying operations
    # Return (strokes_list, new_state) tuple
```

### 5. Register in CLI

Add your renderer to `render.py`:

```python
from .languages.your_language import YourRenderer

# Add to argument choices
parser_single.add_argument("renderer", choices=["deterministic", "probabilistic", "your_language"])

# Add to instantiation logic
elif args.renderer == "your_language":
    renderer = YourRenderer()
```

## Usage

### Command Line Interface

The system provides a CLI through `render.py`:

#### Render Single Program
```bash
python -m renderer.render single <renderer_type> "<program>" <output_file>

# Examples
python -m renderer.render single deterministic "(T l 2 1.5)" output.png
python -m renderer.render single probabilistic "(curve 0.5 2.0 0.3)" curve.png
```

#### Batch Process CSV
```bash  
python -m renderer.render csv <renderer_type> <dataset_name> [--col <column_name>]

# Example
python -m renderer.render csv deterministic spirographs --col program_string
```

**CSV Processing:**
- Reads from: `output/<dataset_name>.csv`
- Saves images to: `output/<dataset_name>/images/{row_index}.png`  
- Writes updated CSV to: `output/<dataset_name>/rendered.csv`

### Python API

```python
from renderer.core import parse_program, render_strokes_to_image, export_image
from renderer.languages.deterministic import DeterministicRenderer

# Parse and evaluate program
renderer = DeterministicRenderer()
ast = parse_program("(T (C l c) 2 0.5)")
strokes = renderer.evaluate(ast)

# Render to image
image_array = render_strokes_to_image(strokes)
export_image(image_array, "output.png")
```

## Configuration

### Rendering Parameters

**Canvas Size**: 512×512 pixels (configurable via `CANVAS_WIDTH_HEIGHT`)  
**Coordinate Bounds**: -5 to +5 in both dimensions (configurable via `XYLIM`)  
**Line Width**: 3.0 pixels (configurable in `render_strokes_to_image()`)

### Probabilistic DSL Parameters

Located in `languages/probabilistic.py`:

```python
NOISE_PARAMS = {
    'sigma': 0.02,        # Parameter noise standard deviation
    'smooth_amp': 0.04,   # Smooth noise amplitude
    'base_freq': 0.5,     # Base noise frequency
    'octaves': 3,         # Noise octaves
    'persistence': 0.5,   # Inter-octave amplitude decay
}
```

## Dependencies

- **numpy**: Numerical computing and array operations
- **cairo**: High-quality 2D graphics rendering
- **imageio**: Image file I/O 
- **pandas**: CSV data processing
- **argparse**: Command-line interface (standard library)

## Examples

### Creating Complex Patterns

**Deterministic Spiral:**
```lisp
(repeat (T l 1 0) 36 (M 1.05 (/ pi 18) 0.1 0))
```

**Probabilistic Tree Branch:**
```lisp
(C (curve 0 2 0.1) 
   (turn -0.5) (curve -0.3 1.5 0.2)
   (lift (* -1 1.5 (cos -0.5)) (* -1 1.5 (sin -0.5)) 0)
   (turn 0.5) (curve 0.3 1.5 0.2))
```

**Colored Composition:**
```lisp
(C (color 1 0 0 (T l 2 0))
   (color 0 1 0 (T l 2 1.57)) 
   (color 0 0 1 (T c 1.5 0)))
```

### Error Handling

The system handles parsing and evaluation errors gracefully:
- **Syntax Errors**: Invalid S-expressions raise `ValueError` with details
- **Unknown Functions**: Undefined operations raise `ValueError` 
- **Batch Processing**: Individual failures are logged but don't stop batch processing
- **File I/O**: Missing directories are created automatically

This architecture provides a solid foundation for experimenting with different DSL designs while maintaining consistent parsing, rendering, and export capabilities.