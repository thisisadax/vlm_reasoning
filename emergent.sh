#!/usr/bin/env bash
set -euo pipefail

OUTROOT="/scratch/gpfs/nb0564/vlm_reasoning/newdata/emergent_trials"
N="${N:-48}"
SEED="${SEED:-1248}"

mkdir -p "$OUTROOT"

# Variety 1: Composition-centric (Schema A)
for K in 2 3 4 5; do
  for V in 0 1 2; do
    if [ "$V" -gt $((K-1)) ]; then continue; fi
    out="${OUTROOT}/A/dims${K}/var${V}"
    mkdir -p "$out"
    python -u generate_emergent_stimuli.py \
      --schema A \
      --outdir "$out" \
      --n-trials "$N" \
      --n-dimensions "$K" \
      --reference-variance "$V" \
      --seed "$SEED"
  done
done

# Variety 2: Motif-centric (Schema B)
for K in 3 4 5; do
  for V in 0 1; do
    if [ "$V" -gt $((K-1)) ]; then continue; fi
    out="${OUTROOT}/B/dims${K}/var${V}"
    mkdir -p "$out"
    python -u generate_emergent_stimuli.py \
      --schema B \
      --outdir "$out" \
      --n-trials "$N" \
      --n-dimensions "$K" \
      --reference-variance "$V" \
      --seed "$SEED"
  done
done

# Variety 3: Color/Symmetry-centric (Schema C)
for K in 2 3 4; do
  for V in 0 1; do
    if [ "$V" -gt $((K-1)) ]; then continue; fi
    out="${OUTROOT}/C/dims${K}/var${V}"
    mkdir -p "$out"
    python -u generate_emergent_stimuli.py \
      --schema C \
      --outdir "$out" \
      --n-trials "$N" \
      --n-dimensions "$K" \
      --reference-variance "$V" \
      --seed "$SEED"
  done
done

echo "✅ Emergent oddball datasets written under: $OUTROOT"
