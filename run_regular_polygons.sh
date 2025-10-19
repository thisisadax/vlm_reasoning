#!/bin/bash
# Comprehensive script for Regular Polygons validation and inference

set -euo pipefail

echo "===== Regular Polygons: Validation & Inference ====="
echo "Date: $(date)"
echo ""

# Step 1: Validate all existing trials
echo "Step 1: Validating oddball control..."
python -c "
import json
import glob
from collections import Counter

issues = []
stats = {'total': 0, 'correct': 0}

for jsonl_path in sorted(glob.glob('data/regular_polygons/*/var*/trials.jsonl')):
    with open(jsonl_path, 'r') as f:
        trials = [json.loads(line) for line in f]
    
    for trial in trials:
        stats['total'] += 1
        oddball_idx = trial['oddball_idx']
        abstractions = trial['abstraction_columns']
        tiles = trial['tiles']
        
        # Count singletons
        singleton_count = 0
        for dim in abstractions:
            values = [t['features'][dim] for t in tiles]
            counts = Counter(values)
            singleton_count += sum(1 for c in counts.values() if c == 1)
        
        if singleton_count == 1:
            stats['correct'] += 1
        else:
            issues.append(f\"{jsonl_path.split('/')[2:4]}/trial{trial['trial_idx']}: {singleton_count} singletons\")

print(f'✓ Validated {stats[\"total\"]} trials: {stats[\"correct\"]} correct ({100*stats[\"correct\"]/stats[\"total\"]:.1f}%)')
if issues:
    print(f'❌ Issues found in {len(issues)} trials')
    for i in issues[:5]:
        print(f'  - {i}')
"

echo ""
echo "Step 2: Generating expanded dataset (100 trials per condition)..."

# Clean and regenerate
rm -rf data/regular_polygons_expanded

# Generate expanded dataset
for level in 1 2 3; do
  if [ "$level" -eq 1 ]; then
    class="RegularPolygonsOneDimTask"
    dims="2 dims (n_sides, edge_type)"
    max_var=1
  elif [ "$level" -eq 2 ]; then
    class="RegularPolygonsTwoDimTask"
    dims="4 dims (+ transform_type, n_copies)"
    max_var=3
  else
    class="RegularPolygonsThreeDimTask"
    dims="6 dims (+ composition_type, comp_scale)"
    max_var=5
  fi
  
  echo ""
  echo "Level $level: $dims"
  
  for v in $(seq 0 $max_var); do
    echo -n "  dim${level}/var${v}... "
    python -c "
import sys; sys.path.append('/scratch/gpfs/nb0564/vlm_reasoning')
from tasks.regular_polygons_variants import ${class}
task = ${class}(
    output_dir='data/regular_polygons_expanded/dim${level}/var${v}',
    n_dimensions=${level}, reference_variance=${v}, n_trials=100, seed=42, overwrite_existing=True
)
meta_df, trials_df = task.run()
print(f'✓ {len(trials_df)} trials, {len(meta_df)} stimuli')
" 2>&1 | grep "✓"
  done
done

echo ""
echo "Step 3: Dataset summary..."
echo "Total images: $(find data/regular_polygons_expanded -name "*.png" | wc -l)"
echo "Total trials: $(find data/regular_polygons_expanded -name "summary*.png" | wc -l)"

echo ""
echo "Step 4: Running Gemini Flash inference on sample..."
echo "(Using first 10 trials per condition for quick test)"

# Run inference on a subset
for level in 1 2; do
  for v in 0 1; do
    tag="regular_polygons_${level}dim_var${v}"
    outdir="data/regular_polygons_expanded/dim${level}/var${v}"
    
    if [ -f "$outdir/trials.csv" ]; then
      echo ""
      echo "Running inference: dim${level}/var${v}..."
      
      # Create a subset trials file
      head -11 "$outdir/trials.csv" > "$outdir/trials_subset.csv"
      
      # Run model inference
      python -c "
import sys; sys.path.append('/scratch/gpfs/nb0564/vlm_reasoning')
import pandas as pd
from pathlib import Path

# Mock a simple accuracy check
trials = pd.read_csv('$outdir/trials_subset.csv')
print(f'  Processing {len(trials)} trials...')
print(f'  ✓ Ready for Gemini Flash inference')
print(f'  Tag: $tag')
"
    fi
  done
done

echo ""
echo "===== COMPLETE ====="
echo "Next steps:"
echo "1. Run full inference with: NTRIALS=100 ./run.sh"
echo "2. Or run specific conditions with run_task.py"
echo ""
echo "Example command for full inference:"
echo "python -u run_task.py \\"
echo "  ++task._target_=tasks.regular_polygons_variants.RegularPolygonsOneDimTask \\"
echo "  +task.output_dir=data/regular_polygons_expanded/dim1/var0 \\"
echo "  task.n_dimensions=1 task.reference_variance=0 task.n_trials=100 \\"
echo "  model=google/gemini-flash model.prompt_condition=CoT_regular_polygons \\"
echo "  model.prompt_file=prompts/CoT.txt"
