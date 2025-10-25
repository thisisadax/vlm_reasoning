#!/usr/bin/env bash
set -euo pipefail

# Generate balanced datasets with NEW inference for four domains.
# Produces:
#  - balanced_data/dim_balanced and balanced_output/dim_balanced (N=TOTAL_PER_DIM per dim 1..3)
#  - balanced_data/var_balanced and balanced_output/var_balanced (N=TOTAL_PER_VAR per var 0..5)
# Balancing rules:
#  - By dim: even across domains and available var buckets at that dim
#  - By var: even across domains and eligible dims for that var
# Remainders are distributed deterministically to reach exact totals.

BASE=${BASE:-"$(pwd)"}
SEED=${SEED:-42}
MODEL=${MODEL:-"google/gemini-flash"}
PROMPT=${PROMPT:-"prompts/CoT.txt"}
TOTAL_PER_DIM=${TOTAL_PER_DIM:-500}
TOTAL_PER_VAR=${TOTAL_PER_VAR:-500}
PYTHON=${PYTHON:-python}
RESUME=${RESUME:-0}

# Ensure env
if [ -f "$HOME/.bashrc" ]; then
  # shellcheck disable=SC1090
  source "$HOME/.bashrc" >/dev/null 2>&1 || true
fi
conda activate fvs >/dev/null 2>&1 || true

BAL_DATA="$BASE/balanced_data"
BAL_OUT="$BASE/balanced_output"
mkdir -p "$BAL_DATA" "$BAL_OUT"

# Build execution plan in TSV: mode\tdomain\tclass\tmodule\tdim\tvar\tn_trials\ttag\tout_rel
PLAN=$("$PYTHON" - <<'PY'
import os, json

TOTAL_PER_DIM = int(os.environ.get('TOTAL_PER_DIM','500'))
TOTAL_PER_VAR = int(os.environ.get('TOTAL_PER_VAR','500'))

# Domain registry: domain -> (module, classes per level 1..3, output subdir under output/)
REG = {
  'regular_polygons': (
    'tasks.regular_polygons_variants',
    {1:'RegularPolygonsOneDimTask',2:'RegularPolygonsTwoDimTask',3:'RegularPolygonsThreeDimTask'},
    'None/gemini-flash',
    lambda d,v: f'regular_polygons_{d}dim_var{v}_dim{d*2}'
  ),
  'glyphs': (
    'tasks.glyphs_variants',
    {1:'GlyphsFirstTask',2:'GlyphsFirstSecondTask',3:'GlyphsAllThreeTask'},
    'glyphs/gemini-flash',
    lambda d,v: f'glyphs_dim{d}_var{v}_dim{d*2}'
  ),
  'nuts_and_bolts': (
    'tasks.nuts_and_bolts_variants',
    {1:'NutsAndBoltsCenterTask',2:'NutsAndBoltsCenterRingTask',3:'NutsAndBoltsBothTask'},
    'nuts_and_bolts/gemini-flash',
    lambda d,v: f'nab_dim{d}_var{v}_dim{d*2}'
  ),
  'totems': (
    'tasks.totems_variants',
    {1:'TotemsOneModuleTask',2:'TotemsTwoModuleTask',3:'TotemsThreeModuleTask'},
    'totems/gemini-flash',
    lambda d,v: f'totems_{d}mod_var{v}_dim{d*2}'
  ),
}

VAR_MAX_BY_DIM = {1:1, 2:3, 3:5}
domains = list(REG.keys())

def distribute(total, buckets):
  base = total // len(buckets)
  rem = total % len(buckets)
  out = {b: base for b in buckets}
  for b in buckets[:rem]:
    out[b] += 1
  return out

rows=[]

# 1) dim_balanced: for each dim, split TOTAL_PER_DIM across (domain,var)
for dim in (1,2,3):
  vars_at_dim = list(range(0, VAR_MAX_BY_DIM[dim]+1))
  # First distribute across domains
  per_domain = distribute(TOTAL_PER_DIM, domains)
  for domain in domains:
    # Then within each domain, split evenly across vars_at_dim
    per_var = distribute(per_domain[domain], vars_at_dim)
    for var in vars_at_dim:
      module, classes, out_rel, tag_build = REG[domain]
      cls = classes[dim]
      tag = tag_build(dim, var)
      rows.append(('dim_balanced', domain, cls, module, dim, var, per_var[var], tag, out_rel))

# 2) var_balanced: for each var, split TOTAL_PER_VAR across (domain,eligible-dims)
for var in range(0,6):
  eligible_dims = [d for d in (1,2,3) if VAR_MAX_BY_DIM[d] >= var]
  buckets = [(domain, dim) for domain in domains for dim in eligible_dims]
  per_bucket = distribute(TOTAL_PER_VAR, buckets)
  for (domain, dim), n in per_bucket.items():
    module, classes, out_rel, tag_build = REG[domain]
    cls = classes[dim]
    tag = tag_build(dim, var)
    rows.append(('var_balanced', domain, cls, module, dim, var, n, tag, out_rel))

for r in rows:
  print('\t'.join(map(str,r)))
PY

# Execute plan
echo "$PLAN" | while IFS=$'\t' read -r MODE DOMAIN CLASS MODULE DIM VAR NTRIALS TAG OUTREL; do
  # Skip zero-sized buckets
  if [ "$NTRIALS" -le 0 ]; then
    continue
  fi
  OUTDIR="$BAL_DATA/$MODE/$DOMAIN/dim${DIM}/var${VAR}"
  mkdir -p "$OUTDIR"
  echo "[RUN] $MODE | $DOMAIN dim${DIM} var${VAR} n=${NTRIALS} -> $OUTDIR"

  # Destination CSV directory
  DST_DIR="$BAL_OUT/$MODE/${OUTREL}"
  mkdir -p "$DST_DIR"
  DST_CSV="$DST_DIR/${TAG}.csv"

  # If resume and final CSV exists, check completeness (rows should equal NTRIALS)
  if [ "$RESUME" -eq 1 ] && [ -f "$DST_CSV" ]; then
    # count data rows (minus header)
    _lines=$(wc -l < "$DST_CSV" | tr -d ' ')
    _rows=$((_lines>0?_lines-1:0))
    if [ "$_rows" -ge "$NTRIALS" ]; then
      echo "  - RESUME: existing CSV complete ($_rows >= $NTRIALS), skipping"
      continue
    else
      echo "  - RESUME: existing CSV incomplete ($_rows < $NTRIALS), will re-run inference"
      rm -f "$DST_CSV" || true
    fi
  fi

  # Generation (skip if resume and trials already exist)
  if [ "$RESUME" -eq 1 ] && [ -s "$OUTDIR/trials.jsonl" ]; then
    _tlines=$(wc -l < "$OUTDIR/trials.jsonl" | tr -d ' ')
    if [ "$_tlines" -ge "$NTRIALS" ]; then
      echo "  - RESUME: trials present ($_tlines >= $NTRIALS), skipping generation"
    else
      echo "  - RESUME: trials incomplete ($_tlines < $NTRIALS), regenerating"
      "$PYTHON" -u run_task.py \
        ++task._target_=${MODULE}.${CLASS} \
        +task.output_dir="$OUTDIR" \
        task.n_dimensions=${DIM} \
        task.reference_variance=${VAR} \
        task.n_trials=${NTRIALS} \
        +task.seed=${SEED} \
        +task.overwrite_existing=true \
        model=${MODEL} \
        model.prompt_condition="${TAG}" \
        model.prompt_file=${PROMPT} \
        +skip_model=true >/dev/null
    fi
  else
    "$PYTHON" -u run_task.py \
      ++task._target_=${MODULE}.${CLASS} \
      +task.output_dir="$OUTDIR" \
      task.n_dimensions=${DIM} \
      task.reference_variance=${VAR} \
      task.n_trials=${NTRIALS} \
      +task.seed=${SEED} \
      +task.overwrite_existing=true \
      model=${MODEL} \
      model.prompt_condition="${TAG}" \
      model.prompt_file=${PROMPT} \
      +skip_model=true >/dev/null || {
        echo "  ! Generation failed for $DOMAIN dim${DIM} var${VAR}; skipping bucket" >&2
        continue
      }
  fi

  # Inference
  if [ "$RESUME" -eq 1 ] && [ -f "$DST_CSV" ]; then
    echo "  - RESUME: final CSV exists, skipping inference"
  else
    # Remove any stale source CSVs before re-running
    SRC_CSV_NONE="$BASE/output/None/gemini-flash/${TAG}.csv"
    SRC_CSV_GLYPHS="$BASE/output/glyphs/gemini-flash/${TAG}.csv"
    SRC_CSV_NAB="$BASE/output/nuts_and_bolts/gemini-flash/${TAG}.csv"
    SRC_CSV_TOTEMS="$BASE/output/totems/gemini-flash/${TAG}.csv"
    rm -f "$SRC_CSV_NONE" "$SRC_CSV_GLYPHS" "$SRC_CSV_NAB" "$SRC_CSV_TOTEMS" 2>/dev/null || true

    "$PYTHON" -u run_task.py \
      ++task._target_=${MODULE}.${CLASS} \
      +task.output_dir="$OUTDIR" \
      task.n_dimensions=${DIM} \
      task.reference_variance=${VAR} \
      task.n_trials=${NTRIALS} \
      +task.seed=${SEED} \
      +task.overwrite_existing=false \
      model=${MODEL} \
      model.prompt_condition="${TAG}" \
      model.prompt_file=${PROMPT} \
      +skip_model=false >/dev/null || {
        echo "  ! Inference failed for $DOMAIN dim${DIM} var${VAR}; skipping move" >&2
        continue
      }
  fi

  # Move CSV to balanced_output
  SRC_CSV_NONE="$BASE/output/None/gemini-flash/${TAG}.csv"
  SRC_CSV_GLYPHS="$BASE/output/glyphs/gemini-flash/${TAG}.csv"
  SRC_CSV_NAB="$BASE/output/nuts_and_bolts/gemini-flash/${TAG}.csv"
  SRC_CSV_TOTEMS="$BASE/output/totems/gemini-flash/${TAG}.csv"

  for C in "$SRC_CSV_NONE" "$SRC_CSV_GLYPHS" "$SRC_CSV_NAB" "$SRC_CSV_TOTEMS"; do
    if [ -f "$C" ]; then
      mv -f "$C" "$DST_DIR/" || true
    fi
  done
done

echo "DONE: balanced_data and balanced_output generated with fresh inference."

