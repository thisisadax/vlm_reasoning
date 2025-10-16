#!/usr/bin/env bash
set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
cd "$PROJECT_ROOT"

OUTROOT="/scratch/gpfs/nb0564/vlm_reasoning/newdata"
NTRIALS="${NTRIALS:-20}"
SEED="${SEED:-1248}"
DUP="${DUP:-8}"
EXCL="${EXCL:-}"     # allow scale if you like
CAP="${CAP:-0}"

run () { python -u run_task.py "$@"; }
cap_flag() { [[ "${CAP}" != "0" ]] && echo "+task.max_stimuli=${CAP}" || true; }

mkdir -p "$OUTROOT"

# ---------- Totems: emergent complexity with transformations and compositions ----------
for complexity in 3 4; do
  for v in 0 1 2; do
    run ++task._target_=tasks.totems_expanded.TotemsEmergentRunner \
        +task.output_dir="${OUTROOT}/expanded/totems/complexity${complexity}/var${v}" \
        task.n_dimensions=3 task.reference_variance=$v task.n_trials=$NTRIALS \
        +task.seed=$SEED +task.dup_factor=$DUP +task.overwrite_existing=true \
        +task.exclude_abstraction_keywords="$EXCL" \
        +task.max_complexity=$complexity +skip_model=true $(cap_flag)
  done
done

# ---------- Glyphs: emergent complexity with transformations and compositions ----------
for complexity in 3 4; do
  for v in 0 1 2 3; do
    run ++task._target_=tasks.glyphs_expanded.GlyphsEmergentRunner \
        +task.output_dir="${OUTROOT}/expanded/glyphs/complexity${complexity}/var${v}" \
        task.n_dimensions=3 task.reference_variance=$v task.n_trials=$NTRIALS \
        +task.seed=$SEED +task.dup_factor=$DUP +task.overwrite_existing=true \
        +task.max_complexity=$complexity +skip_model=true
  done
done

# ---------- Spirographs: emergent complexity with transformations and compositions ----------
for complexity in 3 4; do
  for v in 0 1 2 3; do
    run ++task._target_=tasks.spirographs_expanded.SpirographsEmergentRunner \
        +task.output_dir="${OUTROOT}/expanded/spirographs/complexity${complexity}/var${v}" \
        task.n_dimensions=3 task.reference_variance=$v task.n_trials=$NTRIALS \
        +task.seed=$SEED +task.dup_factor=$DUP +task.overwrite_existing=true \
        +task.max_complexity=$complexity +skip_model=true $(cap_flag)
  done
done

# ---------- Nuts & Bolts: emergent complexity with transformations and compositions ----------
for complexity in 3 4; do
  for v in 0 1 2 3; do
    run ++task._target_=tasks.nuts_and_bolts_expanded.NutsAndBoltsEmergentRunner \
        +task.output_dir="${OUTROOT}/expanded/nuts_and_bolts/complexity${complexity}/var${v}" \
        task.n_dimensions=3 task.reference_variance=$v task.n_trials=$NTRIALS \
        +task.seed=$SEED +task.dup_factor=$DUP +task.overwrite_existing=true \
        +task.max_complexity=$complexity +skip_model=true $(cap_flag)
  done
done

echo "✅ Expanded example generation complete → ${OUTROOT}/expanded"
