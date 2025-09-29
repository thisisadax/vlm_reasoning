import pandas as pd
import itertools
from .base_task import Task
from renderer.languages.relational import RelationalRenderer

class RelationalChainsTask(Task):
    """
    Generates stimuli using the RelationalRenderer with the new
    integer-based indexing system for joins.
    """
    renderer = RelationalRenderer()

    def __init__(self, task_name=None, n_variations_per_program=10, **kwargs):
        """Initializes the RelationalChainsTask."""
        super().__init__(task_name=task_name or "relational_chains", **kwargs)
        self.n_variations_per_program = n_variations_per_program

    def generate_programs(self) -> pd.DataFrame:
        """Generates a DataFrame of programs using index-based joins."""
        records = []
        
        primitives = {
            "line": "(line ? ? ? ?)",
            "circle": "(circle ? ? ?)"
        }
        
        # --- Program Template 1: Hub-and-Spoke Structures ---
        central_hub = "(circle ? ? 1.5)" 
        spoke = primitives["line"]
        # NEW: Join circle's perimeter (index 1) to the line's start (index 0).
        hub_program_str = f"(join {central_hub} {spoke} 1 0)"
        
        records.append({
            "structure_type": "hub_and_spoke",
            "program_string": hub_program_str,
            "chain_composition": "circle-line"
        })

        # --- Program Template 2: Three-Part Chains ---
        prim_types = ["line", "circle"]
        
        for p1_type in prim_types:
            for p2_type in prim_types:
                for p3_type in prim_types:
                    p1_str = primitives[p1_type]
                    p2_str = primitives[p2_type]
                    p3_str = primitives[p3_type]
                    
                    # NEW: All constraints now use integer indices.
                    # The "end" of any primitive is always at index 1.
                    # The "start" (or center) is always at index 0.
                    
                    # 1. Inner join: Connects the "end" (1) of p1 to the "start" (0) of p2.
                    inner_join = f"(join {p1_str} {p2_str} 1 0)"

                    # 2. Outer join: The new composite shape from the inner join has 3 attachment points.
                    #    - Index 0: p1's start/center
                    #    - Index 1: The join point between p1 and p2
                    #    - Index 2: p2's end/perimeter
                    #    We connect this new "end" (index 2) to the "start" (0) of p3.
                    program_string = f"(join {inner_join} {p3_str} 2 0)"
                    
                    records.append({
                        "structure_type": "3_part_chain",
                        "program_string": program_string,
                        "chain_composition": f"{p1_type}-{p2_type}-{p3_type}"
                    })

        # --- Expand the dataset with multiple renderings per program ---
        final_records = []
        for record_template in records:
            for i in range(self.n_variations_per_program):
                new_record = record_template.copy()
                new_record['variation_index'] = i
                final_records.append(new_record)

        print(f"✅ Generated {len(records)} unique program structures.")
        print(f"✅ Created a dataset of {len(final_records)} total stimuli to be rendered.")
        return pd.DataFrame(final_records)

if __name__ == "__main__":
    task = RelationalChainsTask(
        n_variations_per_program=10, 
        n_trials=50
    )
    task.run()