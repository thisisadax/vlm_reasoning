# dsl_renderer/core.py
"""
Core functionality for the DSL renderer system.

This module provides the foundational components for parsing S-expression based 
domain-specific languages (DSLs) and rendering them as geometric graphics. It includes:
- AST node representation for parsed DSL programs
- S-expression parser for converting text programs to AST
- Stroke rendering system using Cairo graphics
- Image export utilities
- CSV batch processing capabilities

The system is designed to be extensible, allowing new DSLs to be added by implementing
the BaseRenderer interface and registering language-specific primitives and functions.
"""
import os
import imageio
import numpy as np
import pandas as pd
import cairo
from typing import List, Union


## --- Core Constants ---
XYLIM = 5.0  # Coordinate bounds for the rendering canvas (-XYLIM to +XYLIM)
CANVAS_WIDTH_HEIGHT = 512  # Output image dimensions in pixels


## --- AST Representation ---
class AstNode:
    """
    A node in the Abstract Syntax Tree representing a DSL program element.
    
    Each AstNode represents either a function call or a primitive in the DSL.
    Function calls have a name and list of arguments, while primitives have
    a name and empty argument list.
    
    Attributes:
        name (str): The function name or primitive identifier
        args (list): List of arguments (AstNodes, floats, or ints)
    
    Examples:
        >>> AstNode('circle', [])  # primitive circle
        >>> AstNode('transform', [AstNode('translate', [1.0, 2.0]), AstNode('circle', [])])
    """
    def __init__(self, name: str, args: list):
        self.name = name
        self.args = args

    def __repr__(self):
        return f"AstNode('{self.name}', {self.args})"


def _atom(token: str) -> Union[int, float, AstNode]:
    """
    Converts a string token into an appropriate data type.
    
    Attempts to parse the token as an integer first, then as a float.
    If neither succeeds, treats it as a symbolic name and creates an AstNode.
    
    Args:
        token: String token to convert
        
    Returns:
        int, float, or AstNode depending on the token content
        
    Examples:
        >>> _atom("42")
        42
        >>> _atom("3.14")
        3.14
        >>> _atom("circle")
        AstNode('circle', [])
    """
    try:
        return int(token)
    except ValueError:
        try:
            return float(token)
        except ValueError:
            # If it's not a number, it's a symbol (e.g., 'l', 'c', 'T')
            return AstNode(token, [])


## --- S-Expression Parser (Shared) ---
def _parse_recursive(tokens: List[str]) -> Union[AstNode, float, int]:
    """
    Recursively parses tokens to build an Abstract Syntax Tree.
    
    This function implements a recursive descent parser for S-expressions.
    It handles parentheses, function calls, and atomic values.
    
    Args:
        tokens: List of string tokens to parse (modified in-place)
        
    Returns:
        AstNode for function calls, or int/float for atomic values
        
    Raises:
        ValueError: If parentheses are unmatched or tokens are malformed
        
    Examples:
        >>> tokens = ['(', 'add', '1', '2', ')']
        >>> _parse_recursive(tokens)
        AstNode('add', [1, 2])
    """
    if not tokens:
        raise ValueError("Unexpected end of program while parsing.")

    token = tokens.pop(0)
    if token == '(':

        # consume all args until ')'
        args = []
        name = tokens.pop(0)  # first item after '(' is the function name
        while tokens[0] != ')':
            args.append(_parse_recursive(tokens))
            if not tokens:
                raise ValueError("Missing ')' in program.")
        
        # pop the closing ')' and return the AstNode
        tokens.pop(0)
        return AstNode(name, args)

    elif token == ')':
        raise ValueError("Unexpected ')' found during parsing.")
    else:
        return _atom(token)


def parse_program(program_string: str) -> AstNode:
    """
    Parses a complete DSL program from S-expression format into an AST.
    
    Takes a string containing an S-expression program and converts it into
    an AstNode tree structure that can be evaluated by a renderer.
    
    Args:
        program_string: Complete S-expression program as a string
        
    Returns:
        AstNode representing the root of the parsed program
        
    Raises:
        ValueError: If the program has syntax errors or is not a valid expression
        
    Examples:
        >>> parse_program("(T l 1.0 0.5)")
        AstNode('T', [AstNode('l', []), 1.0, 0.5])
        >>> parse_program("(C (T l 1 0) (T c 2 1))")
        AstNode('C', [AstNode('T', [AstNode('l', []), 1, 0]), AstNode('T', [AstNode('c', []), 2, 1])])
    """
    s_exp = program_string.replace('(', ' ( ').replace(')', ' ) ')
    tokens = s_exp.split()
    ast = _parse_recursive(tokens)
    if tokens:  # if there are any tokens left, raise an error
        raise ValueError(f"Unexpected tokens at end of program: {' '.join(tokens)}")
    if not isinstance(ast, AstNode):  # if the ast is not an AstNode, raise an error
        raise ValueError("Program must be a valid expression, not just a literal.")
    return ast


def render_strokes_to_image(
    strokes: list,
    canvas_dim: int = CANVAS_WIDTH_HEIGHT,
    coord_bound: float = XYLIM,
    line_width: float = 3.0
) -> np.ndarray:
    """
    Renders a list of geometric strokes to a raster image using Cairo graphics.
    
    Converts stroke data (lists of 2D points) into a rendered PNG image. Supports
    both plain strokes (list of numpy arrays) and colored strokes (tuples of 
    (stroke_array, color)). The coordinate system is automatically scaled from
    the logical bounds to the pixel canvas.
    
    Args:
        strokes: List of strokes. Each stroke can be:
                - numpy array of shape (N, 2) for uncolored strokes
                - tuple of (stroke_array, (r, g, b)) for colored strokes
        canvas_dim: Output image size in pixels (square canvas)
        coord_bound: Logical coordinate bounds (-coord_bound to +coord_bound)
        line_width: Stroke width in pixels
        
    Returns:
        numpy array of shape (canvas_dim, canvas_dim, 3) with RGB values [0,1]
        
    Examples:
        >>> line_stroke = np.array([[0.0, 0.0], [1.0, 1.0]])
        >>> image = render_strokes_to_image([line_stroke])
        >>> image.shape
        (512, 512, 3)
    """
    scale = canvas_dim / (2 * coord_bound)
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, canvas_dim, canvas_dim)
    ctx = cairo.Context(surface)

    # --- Configure Canvas ---
    ctx.set_source_rgb(1, 1, 1) # White background
    ctx.paint()
    ctx.set_line_width(line_width)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    
    default_color = (0.0, 0.0, 0.0) # Default to black

    for stroke in strokes:
        # --- Handle both colored and non-colored DSLs formats ---
        stroke_array, color = stroke if isinstance(stroke, tuple) else (stroke, default_color)
        
        stroke_array = np.asarray(stroke_array)
        if stroke_array.shape[0] < 2:
            continue

        # --- Draw Stroke ---
        ctx.set_source_rgb(*color)
        renderable_stroke = (stroke_array + coord_bound) * scale
        ctx.move_to(renderable_stroke[0, 0], renderable_stroke[0, 1])
        for point in renderable_stroke[1:]:
            ctx.line_to(point[0], point[1])
        ctx.stroke()
        
    # --- Extract Buffer ---
    buf = surface.get_data()
    img_array = np.ndarray(shape=(canvas_dim, canvas_dim, 4), dtype=np.uint8, buffer=buf)
    img_array = img_array[:, :, [2, 1, 0]].astype(np.float32) / 255.0 # Reverse BGRA to RGB
    return img_array


def export_image(image_array: np.ndarray, export_path: str):
    """
    Exports a rendered image array to a PNG file.
    
    Takes a numpy array representing an image and saves it as a PNG file,
    automatically creating the output directory if it doesn't exist.
    
    Args:
        image_array: RGB image as numpy array with values in [0,1] range
        export_path: File path where the PNG should be saved
        
    Raises:
        OSError: If the output directory cannot be created
        
    Examples:
        >>> image = np.random.random((512, 512, 3))
        >>> export_image(image, "output/test.png")
    """
    output_dir = os.path.dirname(export_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    imageio.imwrite(export_path, (image_array * 255).astype(np.uint8))

## --- CSV Processing Utility ---
def render_from_csv(renderer, name: str, program_col: str = "program_string"):
    """
    Batch processes DSL programs from a CSV file and renders them to images.
    
    Reads a CSV file containing DSL programs, renders each program using the
    provided renderer, saves the resulting images, and creates an updated CSV
    with image file paths. Handles errors gracefully by logging them and
    continuing with the next program.
    
    Args:
        renderer: A renderer instance (must implement evaluate() method)
        name: Base name for input CSV file and output directory 
        program_col: Column name containing the DSL program strings
        
    Input:
        - Reads from: output/{name}.csv
        
    Output:
        - Images saved to: output/{name}/images/{row_index}.png
        - Updated CSV saved to: output/{name}/rendered.csv
        
    Raises:
        FileNotFoundError: If the input CSV file doesn't exist
        KeyError: If the specified program column isn't found in the CSV
        
    Examples:
        >>> renderer = DeterministicRenderer()
        >>> render_from_csv(renderer, "spirographs", "program_string")
        # Processes output/spirographs.csv and saves images + updated CSV
    """
    input_csv_path = os.path.join("output", f"{name}.csv")
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input CSV not found: {input_csv_path}")

    df = pd.read_csv(input_csv_path)
    if program_col not in df.columns:
        raise KeyError(f"Column '{program_col}' not found in {input_csv_path}")

    image_output_dir = os.path.join("output", name, "images")
    os.makedirs(image_output_dir, exist_ok=True)

    render_filepaths = []
    for i, row in df.iterrows():
        program_string = str(row[program_col])
        output_path = os.path.join(image_output_dir, f"{i}.png")
        try:
            strokes = renderer.evaluate(parse_program(program_string))
            image_array = render_strokes_to_image(strokes)
            export_image(image_array, output_path)
            render_filepaths.append(output_path)
        except Exception as e:
            print(f"❌ Error processing row {i} ('{program_string[:50]}...'): {e}")
            render_filepaths.append("")

    df["render_filepath"] = render_filepaths
    rendered_csv_path = os.path.join("output", name, "rendered.csv")
    df.to_csv(rendered_csv_path, index=False)
    print(f"\n✅ Wrote updated CSV with filepaths to: {rendered_csv_path}")