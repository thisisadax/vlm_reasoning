#!/usr/bin/env python3
"""
Create comparison summary visualizations across dimensions and variances.

This script creates overview visualizations that compare:
1. The same abstraction type across different dimensions (at fixed variance)
2. The same dimension across different variances (at fixed dimension)
"""

import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import argparse


def find_example_summaries(data_dir: Path, task_name: str = "spirographs"):
    """Find example summary images for each dataset."""
    summaries = {}

    # Look for all spirographs datasets
    for dataset_dir in data_dir.glob(f"{task_name}_dim*_var*"):
        if not dataset_dir.is_dir():
            continue

        # Parse dimension and variance from directory name
        dir_name = dataset_dir.name
        # Format: spirographs_dim{num}_var{num}
        parts = dir_name.split('_')
        if len(parts) >= 3:
            try:
                dim = int(parts[1][3:])  # Extract number after 'dim'
                var = int(parts[2][3:])  # Extract number after 'var'
            except (ValueError, IndexError):
                continue

            summaries_dir = dataset_dir / "summaries"
            if summaries_dir.exists():
                # Find all summary PNG files
                summary_files = list(summaries_dir.glob("*.png"))
                summaries[(dim, var)] = summary_files

    return summaries


def create_dimension_comparison(data_dir: Path, variance: int = 2, output_dir: Path = None, task_name: str = "spirographs"):
    """Create comparison showing same abstraction across different dimensions."""
    if output_dir is None:
        output_dir = data_dir / "comparison_summaries" / task_name
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = find_example_summaries(data_dir, task_name=task_name)
    abstraction_info = get_abstraction_info(data_dir, task_name=task_name)

    # Group by abstraction type (e.g., 'central_primitive', 'central_scale', etc.)
    abstraction_groups = {}

    for (dim, var), files in summaries.items():
        if var != variance:
            continue

        # Group files by abstraction type
        abstraction_files = {}
        for summary_file in files:
            # Parse abstraction name (e.g., 'central_primitive' from 'central_primitive_99' or 'radius' from 'radius_0')
            stem_parts = summary_file.stem.split('_')
            if len(stem_parts) >= 3:
                # Multi-part names like 'central_primitive_0'
                abstraction_name = f"{stem_parts[0]}_{stem_parts[1]}"
            elif len(stem_parts) >= 2:
                # Single-part names like 'radius_0'
                abstraction_name = stem_parts[0]
            else:
                continue

            if abstraction_name not in abstraction_files:
                abstraction_files[abstraction_name] = []
            abstraction_files[abstraction_name].append(summary_file)

        # Pick the first example for each abstraction type (they're all equivalent examples)
        for abstraction_name, file_list in abstraction_files.items():
            if abstraction_name not in abstraction_groups:
                abstraction_groups[abstraction_name] = {}
            # Use the first file as representative
            abstraction_groups[abstraction_name][dim] = file_list[0]

    # Create comparison grids for each abstraction type
    for abstraction, dim_files in abstraction_groups.items():
        if len(dim_files) < 2:
            continue

        # Sort by dimension
        sorted_dims = sorted(dim_files.keys())

        # Load images for each dimension
        images = []
        labels = []
        for dim in sorted_dims:
            if dim in dim_files:
                try:
                    img = Image.open(dim_files[dim])
                    images.append(img)
                    labels.append(f"dim={dim}")
                except Exception as e:
                    print(f"Error loading {dim_files[dim]}: {e}")
                    continue

        if len(images) >= 2:
            # Create informative title
            abstraction_display = abstraction.replace('_', ' ').title()
            title = f"{abstraction_display} Oddball\n(var={variance}, dims {min(sorted_dims)}-{max(sorted_dims)})"
            subtitle = f"This abstraction varies. Other {len(sorted_dims)-1} dims fixed."

            # Create detailed abstraction info
            abstraction_details = format_abstraction_details(abstraction_info, sorted_dims, fixed_var=variance)

            # Create comparison grid with dedicated text panel
            grid = create_comparison_grid(images, labels, title, subtitle, abstraction_details,
                                        is_dimension_comparison=True, varying_dims=sorted_dims,
                                        fixed_var=variance, abstraction_info_dict=abstraction_info)
            if grid:
                output_path = output_dir / f"{abstraction}_var{variance}_dims{min(sorted_dims)}-{max(sorted_dims)}.png"
                grid.save(output_path)
                print(f"Created: {output_path}")


def create_variance_comparison(data_dir: Path, dimensions: list, output_dir: Path = None, task_name: str = "spirographs"):
    """Create comparison showing same variance across different dimensions."""
    if output_dir is None:
        output_dir = data_dir / "comparison_summaries" / task_name
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = find_example_summaries(data_dir, task_name=task_name)
    abstraction_info = get_abstraction_info(data_dir, task_name=task_name)

    # Group by abstraction type and variance
    abstraction_groups = {}

    for (dim, var), files in summaries.items():
        if dim not in dimensions:
            continue

        # Group files by abstraction type for this dimension/variance
        abstraction_files = {}
        for summary_file in files:
            # Parse abstraction name
            stem_parts = summary_file.stem.split('_')
            if len(stem_parts) >= 3:
                # Multi-part names like 'central_primitive_0'
                abstraction_name = f"{stem_parts[0]}_{stem_parts[1]}"
            elif len(stem_parts) >= 2:
                # Single-part names like 'radius_0'
                abstraction_name = stem_parts[0]
            else:
                continue

            if abstraction_name not in abstraction_files:
                abstraction_files[abstraction_name] = []
            abstraction_files[abstraction_name].append(summary_file)

        # Pick the first example for each abstraction type
        for abstraction_name, file_list in abstraction_files.items():
            key = (abstraction_name, var)
            if key not in abstraction_groups:
                abstraction_groups[key] = {}
            # Use the first file as representative
            abstraction_groups[key][dim] = file_list[0]

    # Create comparison grids for each abstraction type and variance
    for (abstraction, var), dim_files in abstraction_groups.items():
        if len(dim_files) < 2:
            continue

        # Sort by dimension
        sorted_dims = sorted(dim_files.keys())

        # Load images for each dimension
        images = []
        labels = []
        for dim in sorted_dims:
            if dim in dim_files:
                try:
                    img = Image.open(dim_files[dim])
                    images.append(img)
                    labels.append(f"dim={dim}")
                except Exception as e:
                    print(f"Error loading {dim_files[dim]}: {e}")
                    continue

        if len(images) >= 2:
            # Create informative title
            abstraction_display = abstraction.replace('_', ' ').title()
            dim_range = f"dims {min(sorted_dims)}-{max(sorted_dims)}" if len(sorted_dims) > 1 else f"dim {sorted_dims[0]}"
            title = f"{abstraction_display} Oddball\n(var={var}, {dim_range})"

            # Create detailed subtitle and abstraction info
            subtitle = f"ODDBALL: {abstraction_display} varies across dimensions"
            abstraction_details = format_abstraction_details(abstraction_info, sorted_dims, fixed_var=var)

            # Create comparison grid with dedicated text panel
            grid = create_comparison_grid(images, labels, title, subtitle, abstraction_details,
                                        is_variance_comparison=True, varying_var=[var],
                                        fixed_dims=dimensions, abstraction_info_dict=abstraction_info)
            if grid:
                dim_suffix = f"{min(sorted_dims)}-{max(sorted_dims)}" if len(sorted_dims) > 1 else str(sorted_dims[0])
                output_path = output_dir / f"{abstraction}_var{var}_dims{dim_suffix}.png"
                grid.save(output_path)
                print(f"Created: {output_path}")


def create_comparison_grid(images, labels, title, subtitle=None, abstraction_info=None, is_dimension_comparison=False, is_variance_comparison=False, varying_dims=None, fixed_var=None, varying_var=None, fixed_dims=None, abstraction_info_dict=None):
    """Create a grid comparing multiple summary images with detailed explanations."""
    if not images:
        return None

    # Assume all images are the same size (they should be from the same format)
    img_width, img_height = images[0].size

    # Calculate grid layout
    n_images = len(images)
    n_cols = min(3, n_images)  # Max 3 columns
    n_rows = (n_images + n_cols - 1) // n_cols

    # Create grid with extended width for dedicated text panel
    title_height = 120  # Increased to prevent title overlap with images
    image_grid_width = n_cols * img_width
    text_panel_width = 800  # Increased space for comprehensive explanations
    grid_width = image_grid_width + text_panel_width
    grid_height = max(n_rows * img_height + title_height, 800)  # Ensure minimum height for text panel

    grid = Image.new('RGB', (grid_width, grid_height), 'white')
    draw = ImageDraw.Draw(grid)

    # Load fonts - very large sizes for readability
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
        header_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        text_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        try:
            title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
            header_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
            text_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
            label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
            small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        except:
            title_font = header_font = text_font = label_font = small_font = ImageFont.load_default()

    # Add title at the top
    title_lines = title.split('\n')
    y_offset = 20
    for line in title_lines:
        bbox = draw.textbbox((0, 0), line, font=title_font)
        text_width = bbox[2] - bbox[0]
        text_x = (image_grid_width - text_width) // 2  # Center in image area only
        draw.text((text_x, y_offset), line, fill='black', font=title_font)
        y_offset += 45

    # Add images
    for i, (img, label) in enumerate(zip(images, labels)):
        row = i // n_cols
        col = i % n_cols

        x = col * img_width
        y = row * img_height + title_height

        grid.paste(img, (x, y))

        # Add label at bottom of image - prominent
        label_bbox = draw.textbbox((0, 0), label, font=label_font)
        label_width = label_bbox[2] - label_bbox[0]
        label_x = x + (img_width - label_width) // 2
        label_y = y + img_height - 30

        draw.text((label_x, label_y), label, fill='red', font=label_font)

    # Draw separator line between image grid and text panel
    separator_x = image_grid_width
    draw.line([(separator_x, 0), (separator_x, grid_height)], fill='gray', width=2)

    # Create comprehensive text panel on the right
    text_x_start = separator_x + 30
    text_y_start = 30
    line_height = 35

    # Extract oddball abstraction from title
    title_lower = title.lower()
    if "central primitive" in title_lower:
        oddball_abstraction = "Central Primitive"
    elif "central scale" in title_lower:
        oddball_abstraction = "Central Scale"
    elif "radial primitive" in title_lower:
        oddball_abstraction = "Radial Primitive"
    elif "radial scale" in title_lower:
        oddball_abstraction = "Radial Scale"
    elif "radius" in title_lower:
        oddball_abstraction = "Radius"
    else:
        oddball_abstraction = "Unknown"

    # Header: What is the oddball?
    draw.text((text_x_start, text_y_start), "🎯 ODDBALL ABSTRACTION", fill='darkred', font=header_font)
    text_y_start += line_height + 10

    draw.text((text_x_start, text_y_start), f"• {oddball_abstraction}", fill='black', font=text_font)
    text_y_start += line_height

    draw.text((text_x_start, text_y_start), "This abstraction varies across conditions", fill='darkblue', font=small_font)
    text_y_start += line_height + 15

    # Header: Experimental parameters
    draw.text((text_x_start, text_y_start), "📊 EXPERIMENTAL PARAMETERS", fill='darkgreen', font=header_font)
    text_y_start += line_height + 10

    if is_dimension_comparison:
        draw.text((text_x_start, text_y_start), f"• Fixed: Variance = {fixed_var}", fill='black', font=text_font)
        text_y_start += line_height
        draw.text((text_x_start, text_y_start), f"• Varying: Dimensions = {varying_dims}", fill='black', font=text_font)
        text_y_start += line_height + 10
    elif is_variance_comparison:
        draw.text((text_x_start, text_y_start), f"• Fixed: Dimensions = {fixed_dims}", fill='black', font=text_font)
        text_y_start += line_height
        draw.text((text_x_start, text_y_start), f"• Varying: Variance = {varying_var}", fill='black', font=text_font)
        text_y_start += line_height + 10

    # Header: Abstractions in each condition
    draw.text((text_x_start, text_y_start), "🔍 ABSTRACTIONS BY CONDITION", fill='darkorange', font=header_font)
    text_y_start += line_height + 10

    if abstraction_info_dict and is_dimension_comparison:
        for dim in varying_dims:
            if (dim, fixed_var) in abstraction_info_dict:
                abs_list = abstraction_info_dict[(dim, fixed_var)]
                abs_str = ", ".join(abs_list)
                draw.text((text_x_start, text_y_start), f"• Dim {dim}:", fill='black', font=text_font)
                text_y_start += line_height

                # Word wrap long abstraction lists
                words = abs_str.split(', ')
                current_line = ""
                for word in words:
                    test_line = current_line + word + ", " if current_line else word + ", "
                    bbox = draw.textbbox((0, 0), test_line, font=small_font)
                    if bbox[2] - bbox[0] > text_panel_width - 80:
                        if current_line:
                            draw.text((text_x_start + 20, text_y_start), current_line.rstrip(', '), fill='darkblue', font=small_font)
                            text_y_start += line_height - 5
                        current_line = word + ", "
                    else:
                        current_line = test_line
                if current_line:
                    draw.text((text_x_start + 20, text_y_start), current_line.rstrip(', '), fill='darkblue', font=small_font)
                    text_y_start += line_height
    elif abstraction_info_dict and is_variance_comparison:
        for var in varying_var:
            abs_sets = []
            for dim in fixed_dims:
                if (dim, var) in abstraction_info_dict:
                    abs_sets.append(set(abstraction_info_dict[(dim, var)]))
            if abs_sets:
                common_abs = sorted(set.intersection(*abs_sets)) if len(abs_sets) > 1 else sorted(abs_sets[0])
                abs_str = ", ".join(common_abs)
                draw.text((text_x_start, text_y_start), f"• Var {var}:", fill='black', font=text_font)
                text_y_start += line_height

                words = abs_str.split(', ')
                current_line = ""
                for word in words:
                    test_line = current_line + word + ", " if current_line else word + ", "
                    bbox = draw.textbbox((0, 0), test_line, font=small_font)
                    if bbox[2] - bbox[0] > text_panel_width - 80:
                        if current_line:
                            draw.text((text_x_start + 20, text_y_start), current_line.rstrip(', '), fill='darkblue', font=small_font)
                            text_y_start += line_height - 5
                        current_line = word + ", "
                    else:
                        current_line = test_line
                if current_line:
                    draw.text((text_x_start + 20, text_y_start), current_line.rstrip(', '), fill='darkblue', font=small_font)
                    text_y_start += line_height

    # Footer explanation
    if text_y_start < grid_height - 100:
        draw.text((text_x_start, text_y_start + 20), "📝 SETUP SUMMARY", fill='purple', font=header_font)
        text_y_start += line_height + 10

        if is_dimension_comparison:
            explanation = f"This comparison shows how {oddball_abstraction.lower()} oddball detection difficulty changes as we add more visual dimensions while keeping reference variance fixed at {fixed_var}."
        elif is_variance_comparison:
            explanation = f"This comparison shows how {oddball_abstraction.lower()} oddball detection difficulty changes as we increase reference variance while keeping dimensions fixed at {fixed_dims}."

        # Word wrap explanation
        words = explanation.split()
        current_line = ""
        for word in words:
            test_line = current_line + " " + word if current_line else word
            bbox = draw.textbbox((0, 0), test_line, font=small_font)
            if bbox[2] - bbox[0] > text_panel_width - 60:
                if current_line:
                    draw.text((text_x_start, text_y_start), current_line, fill='black', font=small_font)
                    text_y_start += line_height - 5
                current_line = word
            else:
                current_line = test_line
        if current_line:
            draw.text((text_x_start, text_y_start), current_line, fill='black', font=small_font)

    return grid


def get_abstraction_info(data_dir: Path, task_name: str = "spirographs"):
    """Get information about abstractions in each dataset."""
    summaries = find_example_summaries(data_dir, task_name=task_name)
    abstraction_info = {}

    for (dim, var), files in summaries.items():
        abstractions = set()
        for summary_file in files[:10]:  # Sample first 10 files
            stem_parts = summary_file.stem.split('_')
            if len(stem_parts) >= 3:
                abstraction_name = f"{stem_parts[0]}_{stem_parts[1]}"
            elif len(stem_parts) >= 2:
                abstraction_name = stem_parts[0]
            else:
                continue
            abstractions.add(abstraction_name)

        abstraction_info[(dim, var)] = sorted(abstractions)

    return abstraction_info


def format_abstraction_details(abstraction_info, varying_dims, fixed_var=None, varying_var=None, fixed_dims=None):
    """Format detailed abstraction information for display."""
    lines = []

    if fixed_var is not None:
        # Variance comparison: same var, different dims
        lines.append(f"Fixed: variance = {fixed_var}")
        lines.append(f"Varying: dimensions = {varying_dims}")

        # Show abstractions for each dimension
        for dim in varying_dims:
            if (dim, fixed_var) in abstraction_info:
                abs_list = abstraction_info[(dim, fixed_var)]
                abs_str = ", ".join(abs_list)
                lines.append(f"Dim {dim} abstractions: {abs_str}")
    else:
        # Dimension comparison: same dims, different vars
        lines.append(f"Fixed: dimensions = {fixed_dims}")
        lines.append(f"Varying: variance = {varying_var}")

        # Show abstractions for each variance
        for var in varying_var:
            abs_sets = []
            for dim in fixed_dims:
                if (dim, var) in abstraction_info:
                    abs_sets.append(set(abstraction_info[(dim, var)]))
            if abs_sets:
                common_abs = sorted(set.intersection(*abs_sets)) if len(abs_sets) > 1 else sorted(abs_sets[0])
                abs_str = ", ".join(common_abs)
                lines.append(f"Var {var} abstractions: {abs_str}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Create comparison summary visualizations')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Data directory containing datasets')
    parser.add_argument('--task', type=str, default='spirographs',
                       help='Task name to process')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for comparison summaries')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = data_dir / "comparison_summaries"

    print(f"Creating comparison summaries from {data_dir}")
    print(f"Output will be saved to {output_dir}")

    # Create variance comparisons with grouped dimensions (avoid crowding)
    print("\nCreating variance comparisons (grouped dimensions)...")
    dimension_groups = [
        [2, 3],      # Group 1: dims 2-3
        [4, 5, 6]    # Group 2: dims 4-6
    ]

    for dimensions in dimension_groups:
        print(f"Processing dimensions {dimensions}...")
        create_variance_comparison(data_dir, dimensions, output_dir, task_name=args.task)

    # Create dimension comparisons (fixed variance, varying dimensions)
    print("\nCreating dimension comparisons (fixed variance)...")
    for variance in [2, 3]:
        print(f"Processing variance {variance}...")
        create_dimension_comparison(data_dir, variance, output_dir, task_name=args.task)

    print(f"\nComparison summaries created in {output_dir}")
    print("\nSummary of what was created:")
    print("- Variance comparisons: Same abstraction across different dimensions at fixed variance")
    print("- Dimension comparisons: Same abstraction across different variances at fixed dimensions")
    print("- Each image shows which abstraction is the 'oddball' and explains the experimental setup")


if __name__ == "__main__":
    main()
