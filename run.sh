#!/usr/bin/env bash
set -euo pipefail

# ----------------- Hydra / experiment knobs -----------------
export PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
cd "$PROJECT_ROOT"

# generation + inference together (Hydra run_task.py does both)
export NTRIALS="${NTRIALS:-25}"
export SEED="${SEED:-1248}"

# model bits (as in your OG setup)
MODEL="${MODEL:-google/gemini-flash}"
PROMPT="${PROMPT:-prompts/CoT.txt}"
DUP="${DUP:-10}"                 # duplication factor inside tasks (kept)
EXCL="${EXCL:-scale}"            # set "" to allow size oddballs
CAP="${CAP:-0}"                  # optional cap if your Task supports +task.max_stimuli

ts="$(date +"%Y%m%d_%H%M%S")"
LOG_DIR="${PROJECT_ROOT}/logs/run_${ts}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "${LOG_DIR}/run.log") 2>&1
set -x

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "NTRIALS=$NTRIALS  SEED=$SEED"
echo "MODEL=$MODEL  PROMPT=$PROMPT"
echo "LOG_DIR=$LOG_DIR"

python - <<'PY'
import os, sys, json, platform
print(json.dumps({
  "python": sys.version,
  "platform": platform.platform(),
  "cwd": os.getcwd(),
  "PROJECT_ROOT": os.environ.get("PROJECT_ROOT"),
  "NTRIALS": os.environ.get("NTRIALS"),
  "SEED": os.environ.get("SEED"),
}, indent=2))
PY

run () { python -u run_task.py "$@"; }

# Check if output directory has completed results
is_completed() {
    local outdir="$1"
    local model="$2"
    local prompt_cond="$3"

    # trials.csv present?
    if [[ ! -f "$outdir/trials.csv" ]]; then
        return 1
    fi

    # model results exist?
    local results_csv="output/${model}/${prompt_cond}.csv"
    if [[ ! -f "$results_csv" ]]; then
        return 1
    fi

    # rows completed?
    local total_trials
    total_trials=$(wc -l < "$outdir/trials.csv")
    local completed_trials
    completed_trials=$(tail -n +2 "$results_csv" | grep -c '^[0-9]')

    if [[ $completed_trials -ge $((total_trials - 1)) ]]; then
        return 0
    else
        return 1
    fi
}

cap_flag() { [[ "${CAP}" != "0" ]] && echo "+task.max_stimuli=${CAP}" || true; }

###############################################################################
# TOTEMS
###############################################################################
# one module (dim1 → var 0..1)
for v in 0 1; do
  tag="totems_1mod_var${v}_dim2"
  outdir="data/totems/dim1/var${v}"
  if is_completed "$outdir" "$MODEL" "$tag"; then
    echo "✅ Skipping completed: $outdir"; continue
  fi
  run \
    ++task._target_=tasks.totems_variants.TotemsOneModuleTask \
    +task.output_dir="$outdir" task.n_dimensions=2 task.reference_variance=$v task.n_trials=$NTRIALS \
    +task.seed=$SEED +task.dup_factor=$DUP +task.overwrite_existing=false +task.exclude_abstraction_keywords="$EXCL" \
    $(cap_flag) \
    model=$MODEL model.prompt_condition=$tag model.prompt_file="$PROMPT"
done

# two modules (dim2 → var 0..3)
for v in 0 1 2 3; do
  tag="totems_2mod_var${v}_dim4"
  outdir="data/totems/dim2/var${v}"
  if is_completed "$outdir" "$MODEL" "$tag"; then
    echo "✅ Skipping completed: $outdir"; continue
  fi
  run \
    ++task._target_=tasks.totems_variants.TotemsTwoModuleTask \
    +task.output_dir="$outdir" task.n_dimensions=4 task.reference_variance=$v task.n_trials=$NTRIALS \
    +task.seed=$SEED +task.dup_factor=$DUP +task.overwrite_existing=false +task.exclude_abstraction_keywords="$EXCL" \
    $(cap_flag) \
    model=$MODEL model.prompt_condition=$tag model.prompt_file="$PROMPT"
done

# three modules (dim3 → var 0..5)
for v in 0 1 2 3 4 5; do
  tag="totems_3mod_var${v}_dim6"
  outdir="data/totems/dim3/var${v}"
  if is_completed "$outdir" "$MODEL" "$tag"; then
    echo "✅ Skipping completed: $outdir"; continue
  fi
  run \
    ++task._target_=tasks.totems_variants.TotemsThreeModuleTask \
    +task.output_dir="$outdir" task.n_dimensions=6 task.reference_variance=$v task.n_trials=$NTRIALS \
    +task.seed=$SEED +task.dup_factor=$DUP +task.overwrite_existing=false +task.exclude_abstraction_keywords="$EXCL" \
    $(cap_flag) \
    model=$MODEL model.prompt_condition=$tag model.prompt_file="$PROMPT"
done

###############################################################################
# NUTS & BOLTS (with inner_phase; top level = 6 dims; var 0..5)
###############################################################################
# center only
for v in 0 1; do
  tag="nab_dim1_var${v}_dim2"
  outdir="data/nuts_and_bolts/dim1/var${v}"
  if is_completed "$outdir" "$MODEL" "$tag"; then
    echo "✅ Skipping completed: $outdir"; continue
  fi
  run \
    ++task._target_=tasks.nuts_and_bolts_variants.NutsAndBoltsCenterTask \
    +task.output_dir="$outdir" task.n_dimensions=2 task.reference_variance=$v task.n_trials=$NTRIALS \
    +task.seed=$SEED +task.dup_factor=$DUP +task.overwrite_existing=false +task.exclude_abstraction_keywords="$EXCL" \
    $(cap_flag) \
    model=$MODEL model.prompt_condition=$tag model.prompt_file="$PROMPT"
done

# center + ring
for v in 0 1 2 3; do
  tag="nab_dim2_var${v}_dim4"
  outdir="data/nuts_and_bolts/dim2/var${v}"
  if is_completed "$outdir" "$MODEL" "$tag"; then
    echo "✅ Skipping completed: $outdir"; continue
  fi
  run \
    ++task._target_=tasks.nuts_and_bolts_variants.NutsAndBoltsCenterRingTask \
    +task.output_dir="$outdir" task.n_dimensions=4 task.reference_variance=$v task.n_trials=$NTRIALS \
    +task.seed=$SEED +task.dup_factor=$DUP +task.overwrite_existing=false +task.exclude_abstraction_keywords="$EXCL" \
    $(cap_flag) \
    model=$MODEL model.prompt_condition=$tag model.prompt_file="$PROMPT"
done

# center + ring + outer (top) → var 0..5  ← UPDATED
for v in 0 1 2 3 4 5; do
  tag="nab_dim3_var${v}_dim6"
  outdir="data/nuts_and_bolts/dim3/var${v}"
  if is_completed "$outdir" "$MODEL" "$tag"; then
    echo "✅ Skipping completed: $outdir"; continue
  fi
  run \
    ++task._target_=tasks.nuts_and_bolts_variants.NutsAndBoltsBothTask \
    +task.output_dir="$outdir" task.n_dimensions=6 task.reference_variance=$v task.n_trials=$NTRIALS \
    +task.seed=$SEED +task.dup_factor=$DUP +task.overwrite_existing=false +task.exclude_abstraction_keywords="$EXCL" \
    $(cap_flag) \
    model=$MODEL model.prompt_condition=$tag model.prompt_file="$PROMPT"
done

###############################################################################
# GLYPHS
###############################################################################
# dim1
for v in 0 1; do
  tag="glyphs_dim1_var${v}_dim2"
  outdir="data/glyphs/dim1/var${v}"
  if is_completed "$outdir" "$MODEL" "$tag"; then
    echo "✅ Skipping completed: $outdir"; continue
  fi
  run \
    ++task._target_=tasks.glyphs_variants.GlyphsFirstTask \
    +task.output_dir="$outdir" task.n_dimensions=2 task.reference_variance=$v task.n_trials=$NTRIALS \
    +task.seed=$SEED +task.dup_factor=$DUP +task.overwrite_existing=false +task.exclude_abstraction_keywords="$EXCL" \
    $(cap_flag) \
    model=$MODEL model.prompt_condition=$tag model.prompt_file="$PROMPT"
done

# dim2
for v in 0 1 2 3; do
  tag="glyphs_dim2_var${v}_dim4"
  outdir="data/glyphs/dim2/var${v}"
  if is_completed "$outdir" "$MODEL" "$tag"; then
    echo "✅ Skipping completed: $outdir"; continue
  fi
  run \
    ++task._target_=tasks.glyphs_variants.GlyphsFirstSecondTask \
    +task.output_dir="$outdir" task.n_dimensions=4 task.reference_variance=$v task.n_trials=$NTRIALS \
    +task.seed=$SEED +task.dup_factor=$DUP +task.overwrite_existing=false +task.exclude_abstraction_keywords="$EXCL" \
    $(cap_flag) \
    model=$MODEL model.prompt_condition=$tag model.prompt_file="$PROMPT"
done

# dim3
for v in 0 1 2 3 4 5; do
  tag="glyphs_dim3_var${v}_dim6"
  outdir="data/glyphs/dim3/var${v}"
  if is_completed "$outdir" "$MODEL" "$tag"; then
    echo "✅ Skipping completed: $outdir"; continue
  fi
  run \
    ++task._target_=tasks.glyphs_variants.GlyphsAllThreeTask \
    +task.output_dir="$outdir" task.n_dimensions=6 task.reference_variance=$v task.n_trials=$NTRIALS \
    +task.seed=$SEED +task.dup_factor=$DUP +task.overwrite_existing=false +task.exclude_abstraction_keywords="$EXCL" \
    $(cap_flag) \
    model=$MODEL model.prompt_condition=$tag model.prompt_file="$PROMPT"
done

###############################################################################
# REGULAR POLYGONS VARIANTS (properly controlled with correct dimensions)
###############################################################################
# Level 1: 2 dims (n_sides, edge_type; var 0..1)
for v in 0 1; do
  tag="regular_polygons_1dim_var${v}_dim2"
  outdir="data/regular_polygons/dim1/var${v}"
  if is_completed "$outdir" "$MODEL" "$tag"; then
    echo "✅ Skipping completed: $outdir"; continue
  fi
  run \
    ++task._target_=tasks.regular_polygons_variants.RegularPolygonsOneDimTask \
    +task.output_dir="$outdir" task.n_dimensions=1 task.reference_variance=$v task.n_trials=$NTRIALS \
    +task.seed=$SEED +task.dup_factor=$DUP +task.overwrite_existing=false +task.exclude_abstraction_keywords="$EXCL" \
    $(cap_flag) \
    model=$MODEL model.prompt_condition=$tag model.prompt_file="$PROMPT"
done

# Level 2: 4 dims (+ transform_type, n_copies; var 0..3)
for v in 0 1 2 3; do
  tag="regular_polygons_2dim_var${v}_dim4"
  outdir="data/regular_polygons/dim2/var${v}"
  if is_completed "$outdir" "$MODEL" "$tag"; then
    echo "✅ Skipping completed: $outdir"; continue
  fi
  run \
    ++task._target_=tasks.regular_polygons_variants.RegularPolygonsTwoDimTask \
    +task.output_dir="$outdir" task.n_dimensions=2 task.reference_variance=$v task.n_trials=$NTRIALS \
    +task.seed=$SEED +task.dup_factor=$DUP +task.overwrite_existing=false +task.exclude_abstraction_keywords="$EXCL" \
    $(cap_flag) \
    model=$MODEL model.prompt_condition=$tag model.prompt_file="$PROMPT"
done

# Level 3: 6 dims (+ composition_type, comp_scale; var 0..5)
for v in 0 1 2 3 4 5; do
  tag="regular_polygons_3dim_var${v}_dim6"
  outdir="data/regular_polygons/dim3/var${v}"
  if is_completed "$outdir" "$MODEL" "$tag"; then
    echo "✅ Skipping completed: $outdir"; continue
  fi
  run \
    ++task._target_=tasks.regular_polygons_variants.RegularPolygonsThreeDimTask \
    +task.output_dir="$outdir" task.n_dimensions=3 task.reference_variance=$v task.n_trials=$NTRIALS \
    +task.seed=$SEED +task.dup_factor=$DUP +task.overwrite_existing=false +task.exclude_abstraction_keywords="$EXCL" \
    $(cap_flag) \
    model=$MODEL model.prompt_condition=$tag model.prompt_file="$PROMPT"
done

echo "=== Done. Logs at: ${LOG_DIR} ==="
