# Regular Polygons Variants: Complete Dimension Matrix

## Task Structure Overview
- **Dimension Levels**: Control complexity of stimuli (D1=2 dims, D2=4 dims, D3=6 dims)
- **Variance Levels**: Control how many dimensions can vary among reference stimuli (0 to max-1)
- **Oddball**: Always exactly ONE dimension is singleton (appears only once across 6 stimuli)

## Base Primitives & Values

### Core Shape Dimensions (D1)
- **n_sides**: [3, 4, 5, 6] - Triangle, Square, Pentagon, Hexagon
- **edge_type**: ['straight', 'convex', 'concave'] - Line edges, outward arcs, inward arcs

### Transformation Dimensions (D2 adds these)
- **transform_type**: ['radial', 'line', 'spiral', 'scale_origin', 'scale_outward']
  - radial: Arrange copies in circle
  - line: Arrange copies in line
  - spiral: Rotating spiral with scaling
  - scale_origin: Concentric scaling from center
  - scale_outward: Exponential scaling with shared anchor (nested quadrants)
- **n_copies**: [3, 4, 5, 6] - Number of copies in transformation

### Composition Dimensions (D3 adds these)
- **composition_type**: ['none', 'nested', 'layered', 'rotated']
  - none: No additional composition
  - nested: Multiple scales nested inside
  - layered: Overlapping at angle
  - rotated: Multiple rotations composed
- **comp_scale**: [0.4, 0.5, 0.6, 0.7] - Scale factor for composition

---

## LEVEL 1: Basic Shapes (2 dimensions total)
**Active Dimensions**: n_sides, edge_type
**Total Combinations**: 4 × 3 = 12 unique stimuli

### dim1/var0: No reference variance
- **Oddball**: ONE of [n_sides, edge_type] differs
- **References**: All 5 references identical
- **Example Trial**:
  - References: 5× (n_sides=4, edge_type=straight) 
  - Oddball: (n_sides=3, edge_type=straight) OR (n_sides=4, edge_type=convex)

### dim1/var1: One dimension varies in references
- **Oddball**: ONE of [n_sides, edge_type] is singleton
- **Other dimension**: Varies among references (at least 2 refs share each value)
- **Example Trial**:
  - Oddball dim: n_sides=3 (appears once)
  - Varying dim: edge_type varies [straight×2, convex×2, concave×1]
  - References mix: (4,straight), (4,convex), (5,straight), (5,convex), (6,concave)
  - Oddball: (3,straight)

---

## LEVEL 2: Shapes with Transformations (4 dimensions total)
**Active Dimensions**: n_sides, edge_type, transform_type, n_copies
**Total Combinations**: 4 × 3 × 5 × 4 = 240 unique stimuli

### dim2/var0: No reference variance
- **Oddball**: ONE of 4 dimensions differs
- **References**: All 5 identical on all 4 dimensions
- **Example Trial**:
  - References: 5× (n_sides=5, edge_type=convex, transform_type=radial, n_copies=4)
  - Oddball: Changes ONE dimension, e.g., (5, convex, spiral, 4)

### dim2/var1: One dimension varies
- **Oddball**: ONE dimension is singleton
- **One other dimension**: Varies among references
- **Two dimensions**: Fixed across all 6 stimuli
- **Example Trial**:
  - Oddball dim: transform_type=spiral (singleton)
  - Varying dim: n_sides varies [3×2, 4×2, 5×1]
  - Fixed: edge_type=straight, n_copies=4 (all stimuli)

### dim2/var2: Two dimensions vary
- **Oddball**: ONE dimension is singleton
- **Two other dimensions**: Vary among references
- **One dimension**: Fixed across all stimuli
- **Example Trial**:
  - Oddball dim: edge_type=concave (singleton)
  - Varying dims: n_sides [3×2, 4×2, 5×1], transform_type [radial×3, line×2]
  - Fixed: n_copies=5 (all stimuli)

### dim2/var3: Three dimensions vary
- **Oddball**: ONE dimension is singleton
- **All three other dimensions**: Vary among references
- **No fixed dimensions** (all 4 can differ)
- **Example Trial**:
  - Oddball dim: n_copies=6 (singleton)
  - Varying: n_sides [3×2, 4×2, 5×1], edge_type [straight×2, convex×2, concave×1], 
            transform_type [radial×2, line×2, spiral×1]

---

## LEVEL 3: Shapes with Transformations and Compositions (6 dimensions total)
**Active Dimensions**: n_sides, edge_type, transform_type, n_copies, composition_type, comp_scale
**Total Combinations**: 4 × 3 × 5 × 4 × 4 × 4 = 3840 unique stimuli

### dim3/var0: No reference variance
- **Oddball**: ONE of 6 dimensions differs
- **References**: All 5 identical on all 6 dimensions
- **Example Trial**:
  - References: 5× (3, straight, radial, 3, nested, 0.5)
  - Oddball: Changes ONE, e.g., (3, straight, radial, 3, layered, 0.5)

### dim3/var1: One dimension varies
- **Oddball**: ONE dimension is singleton
- **One other dimension**: Varies among references
- **Four dimensions**: Fixed across all stimuli
- **Example Trial**:
  - Oddball dim: composition_type=rotated (singleton)
  - Varying dim: n_sides [3×2, 4×2, 5×1]
  - Fixed: edge_type=convex, transform_type=line, n_copies=4, comp_scale=0.6

### dim3/var2: Two dimensions vary
- **Oddball**: ONE dimension is singleton
- **Two other dimensions**: Vary among references
- **Three dimensions**: Fixed across all stimuli
- **Example Trial**:
  - Oddball dim: comp_scale=0.7 (singleton)
  - Varying: n_sides [4×3, 5×2], transform_type [radial×3, spiral×2]
  - Fixed: edge_type=straight, n_copies=3, composition_type=nested

### dim3/var3: Three dimensions vary
- **Oddball**: ONE dimension is singleton
- **Three other dimensions**: Vary among references
- **Two dimensions**: Fixed across all stimuli
- **Example Trial**:
  - Oddball dim: edge_type=concave (singleton)
  - Varying: n_sides [3×2, 4×2, 5×1], n_copies [3×2, 4×2, 5×1], 
            composition_type [none×2, nested×2, layered×1]
  - Fixed: transform_type=scale_origin, comp_scale=0.5

### dim3/var4: Four dimensions vary
- **Oddball**: ONE dimension is singleton
- **Four other dimensions**: Vary among references
- **One dimension**: Fixed across all stimuli
- **Example Trial**:
  - Oddball dim: transform_type=line (singleton)
  - Varying: n_sides [all values appear], edge_type [all values appear],
            n_copies [3×2, 4×2, 5×1], comp_scale [0.4×2, 0.5×2, 0.6×1]
  - Fixed: composition_type=layered

### dim3/var5: Five dimensions vary (maximum variance)
- **Oddball**: ONE dimension is singleton
- **All five other dimensions**: Vary among references
- **No fixed dimensions** (all 6 can differ)
- **Example Trial**:
  - Oddball dim: n_copies=6 (singleton)
  - All others vary: Each dimension has multiple values across references
  - Most complex trials with maximum visual diversity

---

## Key Principles

### Oddball Control Rules
1. **Exactly ONE singleton**: One dimension value appears exactly once (the oddball)
2. **Reference diversity**: Other varying dimensions have ≥2 references sharing each value
3. **No phantom oddballs**: Dimensions that don't affect appearance aren't varied
4. **Balanced sampling**: All dimension values used roughly equally across trials

### Visual Scaling
- **Canvas usage**: ~70% of canvas targeted through dynamic scaling
- **Complexity scaling**: More complex patterns (D3) scaled down to fit
- **Transform scaling**: Line transforms get more spacing, radial gets radius adjustment
- **Composition scaling**: Nested/layered patterns scaled to prevent overlap

### Variance Interpretation
- **var=0**: Maximum control, only oddball differs
- **var=k**: k dimensions vary among references (in addition to oddball)
- **var=max**: Maximum chaos, all non-oddball dimensions can vary

---

## Implementation Details

### Program Generation Flow
1. Base polygon created with (n_sides, edge_type)
2. Transform applied if D2+ with (transform_type, n_copies)
3. Composition applied if D3 with (composition_type, comp_scale)
4. Canvas scaling applied based on complexity

### Renderer Operations
- **Base shapes**: Using PolygonalRenderer with lines (l) and arcs (a)
- **Transforms**: Using repeat operator (6 args: stroke, n, tx, ty, s, theta)
  - scale_outward: Creates exponential scaling (1.4^i for i=0 to n-1)
  - Each shape is 1.4× larger than previous, all anchored at origin
  - Smaller shapes nest perfectly within larger ones (quadrant nesting)
- **Compositions**: Using compose (C) and transform (T) with affine matrices (M)
- **Scaling**: Final transform to fit ~70% canvas with safety margins
  - scale_outward gets aggressive scaling due to exponential growth

### File Structure
```
data/regular_polygons/
├── dim1/
│   ├── var0/  (12 base × 25-100 trials)
│   └── var1/  (12 base × 25-100 trials)
├── dim2/
│   ├── var0/  (192 base × 25-100 trials)
│   ├── var1/
│   ├── var2/
│   └── var3/
└── dim3/
    ├── var0/  (3072 base × 25-100 trials)
    ├── var1/
    ├── var2/
    ├── var3/
    ├── var4/
    └── var5/
```

---

## Testing Checklist
- [x] No double singletons (validated 100% pass)
- [x] Canvas bounds respected (dynamic scaling implemented)
- [x] All shapes rendered (straight, convex, concave polygons)
- [x] All transforms working (radial, line, spiral, scale_origin)
- [x] All compositions working (none, nested, layered, rotated)
- [x] Labels added (red numeric 1-6)
- [x] Gemini Flash inference working (74-99% accuracy observed)
- [x] Proper Hydra integration (run.sh updated)