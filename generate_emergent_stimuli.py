#!/usr/bin/env python3
"""
Emergent-DSL oddball trial generator (SVG + CSV) with 2x3 summary boards.

Now generates 6 stimuli per trial: 5 references + 1 oddball.
All six cells on the 2×3 summary are filled; oddball is boxed.

Schemas define the ordered abstraction dims (presence = first K):

A (composition-centric):
  1) pattern      ∈ {chain, grid, ring, tree}
  2) depth        ∈ {0, 2, 3} (0 when pattern ≠ tree)
  3) n_motifs     ∈ {2,3,4,5,6}
  4) inner_len    ∈ {1,2,3,4}
  5) repeat_n     ∈ {1,3,5}
  6) palette_id   ∈ {0,1,2,3,4}

B (motif-centric):
  1) primitive_bias  ∈ {circrect, polys, mix}
  2) inner_len       ∈ {1,2,3,4}
  3) repeat_n        ∈ {1,3,5}
  4) scale_regime    ∈ {tight, loose}
  5) jitter_regime   ∈ {low, med, high}

C (color/symmetry-centric):
  1) pattern       ∈ {ring, grid}
  2) n_motifs     ∈ {4,5,6}
  3) symmetry_n   ∈ {4,6,8}
  4) palette_id   ∈ {0,1,2,3,4}
  5) inner_len    ∈ {1,2,3}
"""
import argparse
import csv
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any

Number = Union[int, float]
Expr = Union[str, Number, Tuple]

# ---------- DSL ----------
def P(shape: str, size: float) -> Expr:   return ("P", shape, size)
def R(theta: float, e: Expr) -> Expr:     return ("R", theta, e)
def T(dx: float, dy: float, e: Expr) -> Expr: return ("T", dx, dy, e)
def S(sx: float, sy: float, e: Expr) -> Expr: return ("S", sx, sy, e)
def M(axis: str, e: Expr) -> Expr:        return ("M", axis, e)
def C(a: Expr, b: Expr) -> Expr:          return ("C", a, b)
def K(n: int, angle: float, e: Expr) -> Expr: return ("K", n, angle, e)
def COL(r: float, g: float, b: float, a: float, e: Expr) -> Expr:
    return ("COL", r, g, b, a, e)

PRIMS = ("circle","rect","tri","poly5","poly6")
PALETTES = [
    (0.90,0.10,0.10),(0.10,0.60,0.15),(0.12,0.35,0.90),
    (0.80,0.55,0.12),(0.50,0.20,0.80)
]

# ---------- SVG rendering ----------
def _svg_header(w=640,h=640,view=5.0,bg="white") -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="-{view} -{view} {2*view} {2*view}" style="background:{bg}">')

def _poly_points(n: int, size: float):
    return [(size*math.cos(2*math.pi*k/n), size*math.sin(2*math.pi*k/n)) for k in range(n)]

def _expr_to_svg(e: Expr, stack: Optional[List[Tuple[str, Any]]] = None, color=(0,0,0,1.0)) -> List[str]:
    if stack is None: stack=[]
    if not isinstance(e, tuple): return []
    tag=e[0]; out: List[str]=[]
    def tstr():
        acc=[]
        for op in stack:
            if op[0]=="R": acc.append(f"rotate({math.degrees(op[1])})")
            elif op[0]=="T": acc.append(f"translate({op[1]} {op[2]})")
            elif op[0]=="S": acc.append(f"scale({op[1]} {op[2]})")
            elif op[0]=="M":
                ax=op[1]
                if ax=="x": acc.append("scale(1 -1)")
                elif ax=="y": acc.append("scale(-1 1)")
                else: acc.append("matrix(0 1 1 0 0 0)")
        return " ".join(acc)
    if tag=="P":
        _, shape, size = e
        r,g,b,a=color
        stroke=f'stroke="rgba({int(r*255)},{int(g*255)},{int(b*255)},{a})"'
        style=f'fill="none" {stroke} stroke-width="{max(0.05, 0.06*size+0.02)}"'
        trs=tstr()
        if shape=="circle":
            out.append(f'<g transform="{trs}"><circle cx="0" cy="0" r="{size}" {style}/></g>')
        elif shape=="rect":
            s=size; out.append(f'<g transform="{trs}"><rect x="{-s}" y="{-s}" width="{2*s}" height="{2*s}" {style}/></g>')
        elif shape=="tri":
            pts=_poly_points(3,size); d=" ".join(f"{x},{y}" for x,y in pts)
            out.append(f'<g transform="{trs}"><polygon points="{d}" {style}/></g>')
        elif shape.startswith("poly"):
            n=int(shape[4:]); pts=_poly_points(n,size); d=" ".join(f"{x},{y}" for x,y in pts)
            out.append(f'<g transform="{trs}"><polygon points="{d}" {style}/></g>')
        return out
    if tag=="R": _,th,sub=e;   return _expr_to_svg(sub, stack+[("R",th)], color)
    if tag=="T": _,dx,dy,sub=e;return _expr_to_svg(sub, stack+[("T",dx,dy)], color)
    if tag=="S": _,sx,sy,sub=e;return _expr_to_svg(sub, stack+[("S",sx,sy)], color)
    if tag=="M": _,ax,sub=e;   return _expr_to_svg(sub, stack+[("M",ax)], color)
    if tag=="C": _,a,b=e;      return _expr_to_svg(a, stack, color)+_expr_to_svg(b, stack, color)
    if tag=="K":
        _,n,ang,sub=e
        parts=[]
        for i in range(int(n)): parts+=_expr_to_svg(("R",i*ang,sub), stack, color)
        return parts
    if tag=="COL": _,r,g,b,a,sub=e; return _expr_to_svg(sub, stack, (r,g,b,a))
    return []

def render_svg(expr: Expr, w=640, h=640, view=5.0) -> str:
    parts=[_svg_header(w,h,view)]
    parts+=_expr_to_svg(expr)
    parts.append("</svg>")
    return "\n".join(parts)

# ---------- Composition patterns ----------
def compose_chain(motifs: List[Expr]) -> Expr:
    placed=None; spacing=0.9
    for i,m in enumerate(motifs):
        x=-spacing*(len(motifs)-1)/2 + i*spacing
        node=T(x,0.0,m)
        placed=node if placed is None else C(placed,node)
    return placed if placed is not None else P("circle",0.2)

def compose_grid(motifs: List[Expr]) -> Expr:
    placed=None; ncols=max(2,min(4,len(motifs))); cell=1.6
    rows=(len(motifs)+ncols-1)//ncols
    for idx,m in enumerate(motifs):
        r=idx//ncols; c=idx%ncols
        dx=(c-(ncols-1)/2)*cell; dy=(r-(rows-1)/2)*cell
        node=T(dx,dy,m)
        placed=node if placed is None else C(placed,node)
    return placed if placed is not None else P("circle",0.2)

def compose_ring(motifs: List[Expr], symmetry_n: Optional[int]=None) -> Expr:
    placed=None
    n=len(motifs)
    k = symmetry_n if (symmetry_n and symmetry_n>=3) else max(n,3)
    radius=1.6+0.2*max(3,n)
    for i,m in enumerate(motifs):
        ang=i*(2*math.pi/k)
        node=R(ang, T(radius,0.0,m))
        placed=node if placed is None else C(placed,node)
    return placed if placed is not None else P("circle",0.2)

def compose_tree(motifs: List[Expr], depth: int) -> Expr:
    base = motifs[0] if motifs else P("circle",0.2)
    def grow(e: Expr, d: int) -> Expr:
        if d<=0: return e
        L=S(0.88,0.88, R(+0.33, T(-0.9,0.8,e)))
        Rn=S(0.88,0.88, R(-0.33, T(+0.9,0.8,e)))
        return C(grow(L,d-1), grow(Rn,d-1))
    return grow(base, max(1,depth))

# ---------- Motifs ----------
def build_motif(rng: random.Random, inner_len: int, repeat_n: int, palette_id: int,
                primitive_bias: Optional[str]=None,
                scale_regime: Optional[str]=None,
                jitter_regime: Optional[str]=None) -> Expr:
    if primitive_bias=="circrect":
        primitive = rng.choice(["circle","rect"])
    elif primitive_bias=="polys":
        primitive = rng.choice(["poly5","poly6","tri"])
    else:
        primitive = rng.choice(PRIMS)
    if scale_regime=="tight":
        size = rng.uniform(0.25,0.35)
    elif scale_regime=="loose":
        size = rng.uniform(0.18,0.45)
    else:
        size = rng.uniform(0.22,0.40)
    e=P(primitive,size)
    for _ in range(inner_len):
        op=rng.choice(["rot","scale","mirror","jitter"])
        if op=="rot":
            e=R(rng.uniform(-math.pi/6,math.pi/6),e)
        elif op=="scale":
            sx=rng.uniform(0.8,1.2); sy=rng.uniform(0.8,1.2)
            e=S(sx,sy,e)
        elif op=="mirror":
            e=M(rng.choice(["x","y","d"]),e)
        else:
            if jitter_regime=="low": j=0.06
            elif jitter_regime=="high": j=0.18
            else: j=0.12
            e=T(rng.uniform(-j,j), rng.uniform(-j,j), e)
    if repeat_n>1:
        e=K(repeat_n, 2*math.pi/repeat_n, e)
    r,g,b=PALETTES[palette_id%len(PALETTES)]
    e=COL(r,g,b,1.0,e)
    return e

# ---------- Schemas ----------
SCHEMAS = {
    "A": {
        "order": ["pattern","depth","n_motifs","inner_len","repeat_n","palette_id"],
        "values": {
            "pattern":   ["chain","grid","ring","tree"],
            "depth":     [0,2,3],
            "n_motifs":  [2,3,4,5,6],
            "inner_len": [1,2,3,4],
            "repeat_n":  [1,3,5],
            "palette_id":[0,1,2,3,4],
        }
    },
    "B": {
        "order": ["primitive_bias","inner_len","repeat_n","scale_regime","jitter_regime"],
        "values": {
            "primitive_bias": ["circrect","polys","mix"],
            "inner_len":      [1,2,3,4],
            "repeat_n":       [1,3,5],
            "scale_regime":   ["tight","loose"],
            "jitter_regime":  ["low","med","high"],
        }
    },
    "C": {
        "order": ["pattern","n_motifs","symmetry_n","palette_id","inner_len"],
        "values": {
            "pattern":    ["ring","grid"],
            "n_motifs":   [4,5,6],
            "symmetry_n": [4,6,8],
            "palette_id": [0,1,2,3,4],
            "inner_len":  [1,2,3],
        }
    }
}

def sample_assignment(rng: random.Random, order: List[str], values: Dict[str,List]) -> Dict[str,Union[int,str]]:
    a={}
    for d in order:
        a[d]=rng.choice(values[d])
    if "pattern" in a and "depth" in a and a["pattern"]!="tree":
        a["depth"]=0
    return a

def mutate_value(rng: random.Random, dim: str, current, values: Dict[str,List]):
    pool=[v for v in values[dim] if v!=current]
    return rng.choice(pool) if pool else current

# ---------- Builders ----------
def build_program_A(rng: random.Random, a: Dict[str,Union[int,str]]) -> Expr:
    n_motifs  = int(a.get("n_motifs",3))
    inner_len = int(a.get("inner_len",2))
    repeat_n  = int(a.get("repeat_n",1))
    palette   = int(a.get("palette_id",0))
    motifs=[build_motif(rng, inner_len, repeat_n, palette) for _ in range(n_motifs)]
    pattern=a.get("pattern","chain")
    depth=int(a.get("depth",0))
    if pattern=="chain": return compose_chain(motifs)
    if pattern=="grid":  return compose_grid(motifs)
    if pattern=="ring":  return compose_ring(motifs)
    if pattern=="tree":  return compose_tree(motifs, depth=max(1,depth))
    return compose_chain(motifs)

def build_program_B(rng: random.Random, a: Dict[str,Union[int,str]]) -> Expr:
    primitive_bias=a.get("primitive_bias","mix")
    inner_len=int(a.get("inner_len",2))
    repeat_n=int(a.get("repeat_n",1))
    scale_regime=a.get("scale_regime","tight")
    jitter_regime=a.get("jitter_regime","med")
    n_motifs=5
    motifs=[build_motif(rng, inner_len, repeat_n, 0,
                        primitive_bias=primitive_bias,
                        scale_regime=scale_regime,
                        jitter_regime=jitter_regime) for _ in range(n_motifs)]
    return compose_chain(motifs)

def build_program_C(rng: random.Random, a: Dict[str,Union[int,str]]) -> Expr:
    pattern=a.get("pattern","ring")
    n_motifs=int(a.get("n_motifs",5))
    inner_len=int(a.get("inner_len",2))
    palette=int(a.get("palette_id",0))
    symmetry_n=int(a.get("symmetry_n",6))
    motifs=[build_motif(rng, inner_len, 1, palette) for _ in range(n_motifs)]
    if pattern=="ring": return compose_ring(motifs, symmetry_n=symmetry_n)
    else:               return compose_grid(motifs)

BUILDERS = {"A": build_program_A, "B": build_program_B, "C": build_program_C}

def expr_to_svg_file(expr: Expr, path: Path):
    path.write_text(render_svg(expr))

def build_program(schema: str, rng: random.Random, a: Dict[str,Union[int,str]]) -> Expr:
    return BUILDERS[schema](rng, a)

# ---------- Summary board (inline) ----------
def _svg_header_px(width_px: int, height_px: int, W: float, H: float, bg="white") -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_px}" height="{height_px}" '
            f'viewBox="0 0 {W} {H}" style="background:{bg}">')

def _thumb_group(expr: Expr, cx: float, cy: float, w: float, h: float) -> str:
    child_view = 5.0
    sx = (w * 0.85) / (2 * child_view)
    sy = (h * 0.85) / (2 * child_view)
    s = min(sx, sy)
    transform = f'translate({cx} {cy}) scale({s} {s})'
    content = "\n".join(_expr_to_svg(expr, stack=[], color=(0,0,0,1.0)))
    return f'<g transform="{transform}">{content}</g>'

def render_summary_board_inline(
    exprs_by_slot: Dict[int, Expr],
    odd_slot: int,
    legend: str,
) -> str:
    W, H = 24.0, 16.0
    cols, rows = 3, 2
    pad = 1.0
    grid_w = W - 8.0
    cell_w = (grid_w - (cols + 1) * pad) / cols
    cell_h = (H - (rows + 1) * pad) / rows
    grid_x0 = pad
    grid_y0 = pad

    cells = []
    for r in range(rows):
        for c in range(cols):
            x = grid_x0 + c * (cell_w + pad)
            y = grid_y0 + r * (cell_h + pad)
            cx = x + cell_w / 2.0
            cy = y + cell_h / 2.0
            cells.append((x, y, cx, cy))

    snippets: List[str] = []
    for idx, (x, y, cx, cy) in enumerate(cells):
        is_odd = (idx == odd_slot)
        stroke = "#d00" if is_odd else "#000"
        sw = 0.12 if is_odd else 0.08
        snippets.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="none" stroke="{stroke}" stroke-width="{sw}"/>')
        label = str(idx + 1)
        snippets.append(f'<text x="{x + 0.3}" y="{y + 0.9}" font-size="0.9" fill="#333" font-family="monospace">{label}</text>')
        # always a thumbnail now
        expr = exprs_by_slot[idx]
        snippets.append(_thumb_group(expr, cx, cy, cell_w, cell_h))

    # legend
    legend_x = grid_x0 + grid_w + pad
    legend_y = pad
    legend_w = W - legend_x - pad
    legend_h = H - 2 * pad
    snippets.append(f'<rect x="{legend_x}" y="{legend_y}" width="{legend_w}" height="{legend_h}" fill="none" stroke="#333" stroke-width="0.1"/>')
    lines = legend.split("\n")
    ty = legend_y + 1.2
    for line in lines:
        snippets.append(f'<text x="{legend_x + 0.5}" y="{ty}" font-size="0.9" fill="#222" font-family="monospace">{line}</text>')
        ty += 1.0

    svg = [_svg_header_px(1200, 800, W, H)]
    svg += snippets
    svg.append("</svg>")
    return "\n".join(svg)

def summary_legend(schema: str, dims_present: List[str], odd_dim: str,
                   base: Dict[str,Union[int,str]],
                   odd: Dict[str,Union[int,str]],
                   vary_refs: set) -> str:
    lines=[]
    lines.append(f"schema: {schema}")
    lines.append(f"present dims: {', '.join(dims_present)}")
    lines.append(f"oddball_dim: {odd_dim}")
    lines.append("")
    for d in dims_present:
        if d == odd_dim:
            lines.append(f"{d}: REF={base[d]}  ODD={odd[d]}  ← odd")
        else:
            vr = "var" if d in vary_refs else "fixed"
            lines.append(f"{d}: {base[d]}  ({vr} across refs)")
    return "\n".join(lines)

# ---------- Trial synthesis ----------
def make_trial(
    rng: random.Random,
    schema: str,
    order: List[str],
    values: Dict[str,List],
    dims_present: List[str],
    reference_variance: int,
    outdir: Path,
    trial_id: int,
) -> Dict[str,Union[str,int]]:
    base = sample_assignment(rng, dims_present, values)
    odd_dim = rng.choice(dims_present)
    others=[d for d in dims_present if d!=odd_dim]
    rng.shuffle(others)
    vary_refs=set(others[:reference_variance])

    # 5 references now
    ref_assign=[]
    for _ in range(5):
        a=dict(base)
        for d in vary_refs:
            a[d]=mutate_value(rng, d, a[d], values)
        ref_assign.append(a)

    odd=dict(base)
    odd[odd_dim]=mutate_value(rng, odd_dim, base[odd_dim], values)

    tdir = outdir / f"trial_{trial_id:05d}"
    tdir.mkdir(parents=True, exist_ok=True)

    ref_files=[]; ref_exprs=[]
    for i,a in enumerate(ref_assign):
        expr=build_program(schema, rng, a)
        ref_exprs.append(expr)
        fp=tdir / f"ref_{i}.svg"; expr_to_svg_file(expr, fp); ref_files.append(fp.name)
    odd_expr=build_program(schema, rng, odd)
    odd_file="odd.svg"; expr_to_svg_file(odd_expr, tdir/odd_file)

    # randomize placement among 6 slots (0..5): 5 refs + 1 odd
    slots=list(range(6))
    rng.shuffle(slots)
    pos_ref = slots[:5]
    pos_odd = slots[5]

    # build slot -> expr dict for summary
    exprs_by_slot: Dict[int, Expr] = {}
    for i, p in enumerate(pos_ref): exprs_by_slot[p] = ref_exprs[i]
    exprs_by_slot[pos_odd] = odd_expr

    legend = summary_legend(schema, dims_present, odd_dim, base, odd, vary_refs)
    summary_svg = render_summary_board_inline(exprs_by_slot, pos_odd, legend)
    (tdir / "summary.svg").write_text(summary_svg)

    row = {
        "trial_id": trial_id,
        "schema": schema,
        "n_present_dims": len(dims_present),
        "oddball_dim": odd_dim,
        "ref_varied_dims": ",".join(sorted(vary_refs)) if vary_refs else "",
        "summary_svg": "summary.svg",
        "ref0": ref_files[0], "ref1": ref_files[1], "ref2": ref_files[2], "ref3": ref_files[3], "ref4": ref_files[4],
        "odd": odd_file,
        "pos_ref": ",".join(map(str,pos_ref)),
        "pos_odd": str(pos_odd),
    }
    for d in dims_present:
        row[f"{d}_ref"] = base[d]
    row[f"{odd_dim}_odd"] = odd[odd_dim]
    return row

# ---------- CLI ----------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, type=str)
    ap.add_argument("--schema", choices=list(SCHEMAS.keys()), default="A")
    ap.add_argument("--n-trials", type=int, default=48)
    ap.add_argument("--n-dimensions", type=int, default=4)
    ap.add_argument("--reference-variance", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1248)
    args=ap.parse_args()

    rng=random.Random(args.seed)
    schema=args.schema
    order = SCHEMAS[schema]["order"]
    values= SCHEMAS[schema]["values"]
    dims_present = order[: max(1, min(args.n_dimensions, len(order)))]

    max_rv = max(0, len(dims_present)-1)
    if args.reference_variance > max_rv:
        args.reference_variance = max_rv
    if args.reference_variance < 0:
        args.reference_variance = 0

    outdir=Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config.txt").write_text(
        f"schema={schema}\n"
        f"n_trials={args.n_trials}\n"
        f"n_dimensions={len(dims_present)} ({dims_present})\n"
        f"reference_variance={args.reference_variance}\n"
        f"seed={args.seed}\n"
    )

    rows=[]
    for i in range(args.n_trials):
        rows.append(make_trial(
            rng=rng, schema=schema, order=order, values=values,
            dims_present=dims_present, reference_variance=args.reference_variance,
            outdir=outdir, trial_id=i
        ))

    base_cols = ["trial_id","schema","n_present_dims","oddball_dim","ref_varied_dims",
                 "summary_svg","ref0","ref1","ref2","ref3","ref4","odd","pos_ref","pos_odd"]
    dim_cols  = [f"{d}_ref" for d in dims_present] + [f"{d}_odd" for d in dims_present]
    fieldnames = base_cols + list(dict.fromkeys(dim_cols))
    with (outdir / "trials.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for r in rows: w.writerow(r)

    print(f"✅ Wrote {len(rows)} trials to {outdir}")
    print("   - trials.csv")
    print("   - trial_xxxxx/{ref_*.svg, odd.svg, summary.svg}")

if __name__ == "__main__":
    main()
