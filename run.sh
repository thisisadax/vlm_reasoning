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
DUP="${DUP:-10}"                 # duplication factor in task (kept for compatibility)
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

    # Check if trials.csv exists
    if [[ ! -f "$outdir/trials.csv" ]]; then
        return 1  # not completed
    fi

    # Check if results CSV exists and has all trials completed
    local results_csv="output/${model}/${prompt_cond}.csv"
    if [[ ! -f "$results_csv" ]]; then
        return 1  # not completed
    fi

    # Check if all trials have responses
    local total_trials=$(wc -l < "$outdir/trials.csv")
    local completed_trials=$(tail -n +2 "$results_csv" | grep -c '^[0-9]')

    if [[ $completed_trials -ge $((total_trials - 1)) ]]; then  # -1 for header
        return 0  # completed
    else
        return 1  # not completed
    fi
}
cap_flag() { [[ "${CAP}" != "0" ]] && echo "+task.max_stimuli=${CAP}" || true; }

###############################################################################
# SPIROGRAPHS
###############################################################################
# central-only
for v in 0 1; do
  tag="spiro_central_var${v}_dim2"
  outdir="data/spirographs/dim1/var${v}"
  if is_completed "$outdir" "$MODEL" "$tag"; then
    echo "✅ Skipping completed: $outdir"
    continue
  fi
  run \
    ++task._target_=tasks.spirographs_variants.SpirographsCentralTask \
    +task.output_dir="$outdir" task.n_dimensions=2 task.reference_variance=$v task.n_trials=$NTRIALS \
    +task.seed=$SEED +task.dup_factor=$DUP +task.overwrite_existing=false +task.exclude_abstraction_keywords="$EXCL" \
    $(cap_flag) \
    model=$MODEL model.prompt_condition=$tag model.prompt_file="$PROMPT"
done

# radial-only
for v in 0 1 2; do
  tag="spiro_radial_var${v}_dim3"
  outdir="data/spirographs/dim2/var${v}"
  if is_completed "$outdir" "$MODEL" "$tag"; then
    echo "✅ Skipping completed: $outdir"
    continue
  fi
  run \
    ++task._target_=tasks.spirographs_variants.SpirographsRadialTask \
    +task.output_dir="$outdir" task.n_dimensions=3 task.reference_variance=$v task.n_trials=$NTRIALS \
    +task.seed=$SEED +task.dup_factor=$DUP +task.overwrite_existing=false +task.exclude_abstraction_keywords="$EXCL" \
    $(cap_flag) \
    model=$MODEL model.prompt_condition=$tag model.prompt_file="$PROMPT"
done

# both
for v in 0 1 2 3 4; do
  tag="spiro_both_var${v}_dim5"
  outdir="data/spirographs/dim3/var${v}"
  if is_completed "$outdir" "$MODEL" "$tag"; then
    echo "✅ Skipping completed: $outdir"
    continue
  fi
  run \
    ++task._target_=tasks.spirographs_variants.SpirographsBothTask \
    +task.output_dir="$outdir" task.n_dimensions=5 task.reference_variance=$v task.n_trials=$NTRIALS \
    +task.seed=$SEED +task.dup_factor=$DUP +task.overwrite_existing=false +task.exclude_abstraction_keywords="$EXCL" \
    $(cap_flag) \
    model=$MODEL model.prompt_condition=$tag model.prompt_file="$PROMPT"
done

###############################################################################
# TOTEMS
###############################################################################
# one module
for v in 0 1; do
  tag="totems_1mod_var${v}_dim2"
  outdir="data/totems/dim1/var${v}"
  if is_completed "$outdir" "$MODEL" "$tag"; then
    echo "✅ Skipping completed: $outdir"
    continue
  fi
  run \
    ++task._target_=tasks.totems_variants.TotemsOneModuleTask \
    +task.output_dir="$outdir" task.n_dimensions=2 task.reference_variance=$v task.n_trials=$NTRIALS \
    +task.seed=$SEED +task.dup_factor=$DUP +task.overwrite_existing=false +task.exclude_abstraction_keywords="$EXCL" \
    model=$MODEL model.prompt_condition=$tag model.prompt_file="$PROMPT"
done

# two modules
for v in 0 1 2 3; do
  tag="totems_2mod_var${v}_dim4"
  outdir="data/totems/dim2/var${v}"
  if is_completed "$outdir" "$MODEL" "$tag"; then
    echo "✅ Skipping completed: $outdir"
    continue
  fi
  run \
    ++task._target_=tasks.totems_variants.TotemsTwoModuleTask \
    +task.output_dir="$outdir" task.n_dimensions=4 task.reference_variance=$v task.n_trials=$NTRIALS \
    +task.seed=$SEED +task.dup_factor=$DUP +task.overwrite_existing=false +task.exclude_abstraction_keywords="$EXCL" \
    model=$MODEL model.prompt_condition=$tag model.prompt_file="$PROMPT"
done

# three modules
for v in 0 1 2 3 4 5; do
  tag="totems_3mod_var${v}_dim6"
  outdir="data/totems/dim3/var${v}"
  if is_completed "$outdir" "$MODEL" "$tag"; then
    echo "✅ Skipping completed: $outdir"
    continue
  fi
  run \
    ++task._target_=tasks.totems_variants.TotemsThreeModuleTask \
    +task.output_dir="$outdir" task.n_dimensions=6 task.reference_variance=$v task.n_trials=$NTRIALS \
    +task.seed=$SEED +task.dup_factor=$DUP +task.overwrite_existing=false +task.exclude_abstraction_keywords="$EXCL" \
    model=$MODEL model.prompt_condition=$tag model.prompt_file="$PROMPT"
done

###############################################################################
# NUTS & BOLTS
###############################################################################
# dim1
for v in 0 1; do
  tag="nab_dim1_var${v}_dim2"
  outdir="data/nuts_and_bolts/dim1/var${v}"
  if is_completed "$outdir" "$MODEL" "$tag"; then
    echo "✅ Skipping completed: $outdir"
    continue
  fi
  run \
    ++task._target_=tasks.nuts_and_bolts_variants.NutsAndBoltsCenterTask \
    +task.output_dir="$outdir" task.n_dimensions=2 task.reference_variance=$v task.n_trials=$NTRIALS \
    +task.seed=$SEED +task.dup_factor=$DUP +task.overwrite_existing=false +task.exclude_abstraction_keywords="$EXCL" \
    $(cap_flag) \
    model=$MODEL model.prompt_condition=$tag model.prompt_file="$PROMPT"
done

# dim2
for v in 0 1 2 3; do
  tag="nab_dim2_var${v}_dim4"
  outdir="data/nuts_and_bolts/dim2/var${v}"
  if is_completed "$outdir" "$MODEL" "$tag"; then
    echo "✅ Skipping completed: $outdir"
    continue
  fi
  run \
    ++task._target_=tasks.nuts_and_bolts_variants.NutsAndBoltsCenterRingTask \
    +task.output_dir="$outdir" task.n_dimensions=4 task.reference_variance=$v task.n_trials=$NTRIALS \
    +task.seed=$SEED +task.dup_factor=$DUP +task.overwrite_existing=false +task.exclude_abstraction_keywords="$EXCL" \
    $(cap_flag) \
    model=$MODEL model.prompt_condition=$tag model.prompt_file="$PROMPT"
done

# dim3
for v in 0 1 2 3 4; do
  tag="nab_dim3_var${v}_dim5"
  outdir="data/nuts_and_bolts/dim3/var${v}"
  if is_completed "$outdir" "$MODEL" "$tag"; then
    echo "✅ Skipping completed: $outdir"
    continue
  fi
  run \
    ++task._target_=tasks.nuts_and_bolts_variants.NutsAndBoltsBothTask \
    +task.output_dir="$outdir" task.n_dimensions=5 task.reference_variance=$v task.n_trials=$NTRIALS \
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
    echo "✅ Skipping completed: $outdir"
    continue
  fi
  run \
    ++task._target_=tasks.glyphs_variants.GlyphsFirstTask \
    +task.output_dir="$outdir" task.n_dimensions=2 task.reference_variance=$v task.n_trials=$NTRIALS \
    +task.seed=$SEED +task.dup_factor=$DUP +task.overwrite_existing=false +task.exclude_abstraction_keywords="$EXCL" \
    model=$MODEL model.prompt_condition=$tag model.prompt_file="$PROMPT"
done

# dim2
for v in 0 1 2 3; do
  tag="glyphs_dim2_var${v}_dim4"
  outdir="data/glyphs/dim2/var${v}"
  if is_completed "$outdir" "$MODEL" "$tag"; then
    echo "✅ Skipping completed: $outdir"
    continue
  fi
  run \
    ++task._target_=tasks.glyphs_variants.GlyphsFirstSecondTask \
    +task.output_dir="$outdir" task.n_dimensions=4 task.reference_variance=$v task.n_trials=$NTRIALS \
    +task.seed=$SEED +task.dup_factor=$DUP +task.overwrite_existing=false +task.exclude_abstraction_keywords="$EXCL" \
    model=$MODEL model.prompt_condition=$tag model.prompt_file="$PROMPT"
done

# dim3
for v in 0 1 2 3 4 5; do
  tag="glyphs_dim3_var${v}_dim6"
  outdir="data/glyphs/dim3/var${v}"
  if is_completed "$outdir" "$MODEL" "$tag"; then
    echo "✅ Skipping completed: $outdir"
    continue
  fi
  run \
    ++task._target_=tasks.glyphs_variants.GlyphsAllThreeTask \
    +task.output_dir="$outdir" task.n_dimensions=6 task.reference_variance=$v task.n_trials=$NTRIALS \
    +task.seed=$SEED +task.dup_factor=$DUP +task.overwrite_existing=false +task.exclude_abstraction_keywords="$EXCL" \
    model=$MODEL model.prompt_condition=$tag model.prompt_file="$PROMPT"
done

echo "=== Done. Logs at: ${LOG_DIR} ==="
