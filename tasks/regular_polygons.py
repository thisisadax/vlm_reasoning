# tasks/regular_polygons.py
import pandas as pd
import itertools
import math
from renderer.languages.polygonal import PolygonalRenderer
from tasks.base_task import Task


class PolygonalTask(Task):
    """
    A program generation task for creating a wide variety of shapes using
    a single, unified generation loop for chained primitive operations.
    
    This task generates both centered regular polygons and more complex open
    shapes (like spirals) by systematically varying the parameters of the
    'repeat' operator. Polygons are treated as a special case where translation
    is zero and scaling is one.
    """
    # --- specify the new renderer for the task ---
    renderer = PolygonalRenderer()

    def __init__(self, task_name=None, **kwargs):
        """Initializes the PolygonalTask."""
        super().__init__(task_name=task_name or "polygonal", **kwargs)

    def generate_programs(self):
        """
        Generates a DataFrame of programs using a unified loop.
        
        This method iterates through a grid of parameters for n_repeats,
        translation, and scaling. It conditionally applies a centering
        transformation for parameter combinations that result in closed,
        regular polygons.
        """
        records = []
        curviness_magnitude = 0.25

        # Define a single, comprehensive hyperparameter grid
        param_grid = {
            "n_repeats": range(3, 8),
            "side_type": ["straight", "concave"],
            "t": [0, 0.25],
            "scale_factor": [1.0, 1.25, 1.5],
        }

        # Create all unique combinations of parameters
        keys, values = zip(*param_grid.items())
        all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        print(f"Generating programs from {len(all_combinations)} parameter combinations...")

        for params in all_combinations:
            n_repeats = params['n_repeats']
            side_type = params['side_type']
            tx = params['t']
            ty = params['t']
            s = params['scale_factor']

            # Define the geometric primitive based on side_type
            if side_type == "straight":
                primitive = "l"
                curviness = 0.0
            else:
                primitive = f"(a {curviness_magnitude})"
                curviness = curviness_magnitude
            
            # The angle of rotation is determined by n_repeats to maintain rotational symmetry
            angle = (2 * math.pi) / n_repeats
            
            # Build the base program string
            base_program = f"(repeat {primitive} {n_repeats} {tx} {ty} {s} {angle:.5f})"
            final_program = base_program

            # Calculate the apothem for a regular polygon with side length 1
            apothem = 1 / (2 * math.tan(math.pi / n_repeats))
            center_x = -0.5
            center_y = -apothem
            final_program = f"(T {base_program} (M 1 0 {center_x:.5f} {center_y:.5f}))"
            
            records.append({
                "n_repeats": n_repeats,
                "side_type": side_type,
                "curviness": curviness,
                "tx": tx,
                "ty": ty,
                "scale_factor": s,
                "angle": angle,
                "program_string": final_program,
            })
            
        df = pd.DataFrame(records)
        print(f"✅ Generated {len(df)} total unique programs.")
        return df


if __name__ == "__main__":
    task = PolygonalTask()
    task.run()