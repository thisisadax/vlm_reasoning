# dsl_renderer/render.py
import argparse
from .core import (
    parse_program, render_strokes_to_image, export_image, render_from_csv
)
from .languages.deterministic import DeterministicRenderer
from .languages.probabilistic import ProbabilisticRenderer


def main():
    """Main execution function with command-line parsing."""
    parser = argparse.ArgumentParser(
        description="A unified renderer for deterministic and probabilistic DSLs.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- Parser for rendering a single program ---
    parser_single = subparsers.add_parser("single", help="Render a single program string.")
    parser_single.add_argument("renderer", choices=["deterministic", "probabilistic"], help="The type of DSL renderer to use.")
    parser_single.add_argument("program", type=str, help="The program string to render (in S-expression format).")
    parser_single.add_argument("output", type=str, help="The path to save the output PNG image.")

    # --- Parser for rendering from a CSV file ---
    parser_csv = subparsers.add_parser("csv", help="Render all programs from a CSV file.")
    parser_csv.add_argument("renderer", choices=["deterministic", "probabilistic"], help="The type of DSL renderer to use.")
    parser_csv.add_argument("name", type=str, help="Base name of the CSV in 'output/' (e.g., 'spirographs').")
    parser_csv.add_argument("--col", type=str, default="program_string", help="Column with S-expression programs.")

    args = parser.parse_args()

    # --- Instantiate the selected renderer ---
    if args.renderer == "deterministic":
        renderer = DeterministicRenderer()
    elif args.renderer == "probabilistic":
        renderer = ProbabilisticRenderer()
    else:
        raise ValueError("Invalid renderer type specified.")

    # --- Execute the chosen command ---
    if args.command == "single":
        print(f"Rendering single '{args.renderer}' program...")
        ast = parse_program(args.program)
        strokes = renderer.evaluate(ast)
        image_array = render_strokes_to_image(strokes)
        export_image(image_array, args.output)

    elif args.command == "csv":
        print(f"Rendering CSV '{args.name}.csv' with '{args.renderer}' renderer...")
        render_from_csv(renderer, args.name, program_col=args.col)

if __name__ == "__main__":
    main()