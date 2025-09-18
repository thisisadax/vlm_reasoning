# dsl_renderer/core.py
import os
import imageio
import numpy as np
import pandas as pd
import cairo
from typing import List, Union


## --- Core Constants ---
XYLIM = 5.0
CANVAS_WIDTH_HEIGHT = 512


## --- AST Representation ---
class AstNode:
    """A simple node to represent the DSL's Abstract Syntax Tree."""
    def __init__(self, name: str, args: list):
        self.name = name
        self.args = args

    def __repr__(self):
        return f"AstNode('{self.name}', {self.args})"


def _atom(token: str) -> Union[int, float, AstNode]:
    """Converts a token into an int, float, or symbolic AstNode."""
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
    """Recursively consumes tokens to build the AST."""
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
    """Parses a program string in S-expression format into an AstNode structure."""
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
    Renders a list of strokes to a numpy image array, elegantly handling
    both colored (tuple) and non-colored (array) stroke data.
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
    """Saves a NumPy array as a PNG image."""
    output_dir = os.path.dirname(export_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    imageio.imwrite(export_path, (image_array * 255).astype(np.uint8))

## --- CSV Processing Utility ---
def render_from_csv(renderer, name: str, program_col: str = "program_string"):
    """
    Loads a CSV, renders programs using the provided renderer, and saves results.
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