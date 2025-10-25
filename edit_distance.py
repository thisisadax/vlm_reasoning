#!/usr/bin/env python3
import argparse, csv, json, re, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Optional: enable TREE distance if zss + stitch_core are available
TRY_TREE = True
TREE_READY = False
if TRY_TREE:
    try:
        import zss  # pip install zss
        from stitch_core import parse as stitch_parse  # pip install stitch_core
        TREE_READY = True
    except Exception:
        TREE_READY = False

BASE = Path("/scratch/gpfs/nb0564/vlm_reasoning").resolve()

@dataclass
class Bucket:
    mode: str    # "data" | "dim_balanced" | "var_balanced"
    domain: str  # "regular_polygons" | "glyphs" | "nuts_and_bolts" | "totems"
    dim: int
    var: int
    path: Path   # folder containing metadata.csv, trials.csv, trials/

# ------------------------------ utilities ------------------------------

def tokenize(program: str) -> List[str]:
    """
    Split a Stitch program string into lightweight tokens:
    parentheses as tokens + non-space runs.
    """
    return re.findall(r"[()]|[^\s()]+", program.strip())

def levenshtein_tokens(a_tokens: List[str], b_tokens: List[str]) -> int:
    """
    Standard O(n*m) DP Levenshtein over token sequences.
    """
    n, m = len(a_tokens), len(b_tokens)
    if n == 0: return m
    if m == 0: return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            ins = dp[j - 1] + 1
            delete = dp[j] + 1
            subst = prev + (0 if a_tokens[i - 1] == b_tokens[j - 1] else 1)
            prev, dp[j] = dp[j], min(ins, delete, subst)
    return dp[m]

# ---- optional: tree distance via stitch_core.parse + zss ----

class _Node:
    __slots__ = ("label", "children")
    def __init__(self, label: str, children: Optional[List["__class__"]] = None):
        self.label = label
        self.children = children or []
    def get_children(self): return self.children
    def get_label(self): return self.label

def _sexpr_to_tree(sexpr) -> _Node:
    """
    Convert stitch_core.parse S-expr (lists/atoms) into a simple tree for zss.
    We treat a list like [head, arg1, arg2, ...] with head as the node label.
    Atoms become leaf nodes.
    """
    if isinstance(sexpr, (str, int, float)):
        return _Node(str(sexpr))
    if isinstance(sexpr, list) and len(sexpr) > 0:
        head = str(sexpr[0])
        kids = [_sexpr_to_tree(x) for x in sexpr[1:]]
        return _Node(head, kids)
    # empty or unknown → generic node
    return _Node("∅")

def tree_edit_distance(a_prog: str, b_prog: str) -> int:
    a_tree = _sexpr_to_tree(stitch_parse(a_prog))
    b_tree = _sexpr_to_tree(stitch_parse(b_prog))
    return zss.distance(a_tree, b_tree, get_children=_Node.get_children, get_label=_Node.get_label)

def token_edit_distance(a_prog: str, b_prog: str) -> int:
    return levenshtein_tokens(tokenize(a_prog), tokenize(b_prog))

# ------------------------------ I/O helpers ------------------------------

def find_buckets(domains: List[str], roots: List[Path]) -> List[Bucket]:
    buckets: List[Bucket] = []
    for root in roots:
        if not root.exists(): continue
        for domain in domains:
            for mode, pattern in [
                ("data", f"{domain}/dim*/var*"),
                ("dim_balanced", f"dim_balanced/{domain}/dim*/var*"),
                ("var_balanced", f"var_balanced/{domain}/dim*/var*"),
            ]:
                for var_dir in root.glob(pattern):
                    if not var_dir.is_dir(): continue
                    try:
                        dim = int(var_dir.parent.name.replace("dim", ""))
                        var = int(var_dir.name.replace("var", ""))
                    except Exception:
                        continue
                    if (var_dir / "metadata.csv").exists() and (var_dir / "trials.csv").exists() and (var_dir / "trials").exists():
                        buckets.append(Bucket(mode, domain, dim, var, var_dir))
    return buckets

# Build features→program map (per-bucket) using metadata.csv
EXCLUDE_META_COLS = {"program_string", "render_filepath", "stimulus_idx", "id", "file", "filepath"}

def _row_to_key(row: pd.Series) -> Tuple:
    items = []
    for k, v in row.items():
        if k in EXCLUDE_META_COLS: continue
        # normalize numbers (int if int-like, else float), keep strings as-is
        try:
            if isinstance(v, str) and v.strip() == "":
                vv = ""
            elif float(v).is_integer():
                vv = int(float(v))
            else:
                vv = float(v)
        except Exception:
            vv = str(v)
        items.append((k, vv))
    return tuple(sorted(items))

def build_features_to_program(meta_csv: Path) -> Dict[Tuple, str]:
    df = pd.read_csv(meta_csv)
    if "program_string" not in df.columns:
        return {}
    f2p: Dict[Tuple, str] = {}
    for _, row in df.iterrows():
        prog = str(row["program_string"]).strip()
        if not prog: continue
        f2p[_row_to_key(row)] = prog
    return f2p

def read_trial_meta(trials_dir: Path, trial_idx: int) -> Optional[dict]:
    # expected: trials/trial=<idx>_meta.json
    p = trials_dir / f"trial={trial_idx}_meta.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

def get_tile_features(meta: dict) -> List[dict]:
    """
    Expect either meta['tiles'][i]['features'] or meta['tiles'][i] has 'features'.
    """
    tiles = meta.get("tiles", [])
    feats = []
    for t in tiles:
        feats.append(t.get("features", t))
    return feats

def oddball_zero_based(oddball_idx_1based: int) -> int:
    return max(0, int(oddball_idx_1based) - 1)

# ------------------------------ main computation ------------------------------

def compute_per_trial(buckets: List[Bucket], distance_mode: str, out_csv: Path, limit: Optional[int] = None):
    if distance_mode == "tree":
        if not TREE_READY:
            print("[warn] TREE distance requested, but zss/stitch_core not available. Falling back to TOKEN.")
            dist = token_edit_distance
        else:
            dist = tree_edit_distance
    else:
        dist = token_edit_distance  # default

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f_out:
        w = csv.writer(f_out)
        w.writerow([
            "mode", "domain", "dim", "var", "trial_idx", "oddball_idx",
            "n_tiles", "avg_edit_distance", "distances_json"
        ])

        for b in buckets:
            meta_csv = b.path / "metadata.csv"
            trials_csv = b.path / "trials.csv"
            trials_dir = b.path / "trials"

            f2p = build_features_to_program(meta_csv)
            if not f2p:
                print(f"[SKIP] no program_string in {meta_csv}")
                continue

            tdf = pd.read_csv(trials_csv)
            if "trial_idx" not in tdf.columns or "oddball_idx" not in tdf.columns:
                print(f"[SKIP] missing trial_idx/oddball_idx in {trials_csv}")
                continue

            print(f"[RUN] {b.mode}/{b.domain} dim{b.dim} var{b.var} | trials={len(tdf)} | map={len(f2p)}")
            for _, row in tdf.iterrows():
                t_idx = int(row["trial_idx"])
                odd1 = int(row["oddball_idx"])
                meta = read_trial_meta(trials_dir, t_idx)
                if meta is None:  # no sidecar
                    continue
                feats = get_tile_features(meta)
                if not feats:
                    continue

                # Resolve programs per tile via features→program
                progs: List[Optional[str]] = []
                for f in feats:
                    key = tuple(sorted([(k, f[k]) for k in f.keys()]))
                    # if metadata.csv uses slightly different numeric types, normalize via EXCLUDE_META_COLS logic
                    # build a key shaped like _row_to_key outputs:
                    norm_items = []
                    for k, v in f.items():
                        if k in EXCLUDE_META_COLS: continue
                        try:
                            if float(v).is_integer():
                                vv = int(float(v))
                            else:
                                vv = float(v)
                        except Exception:
                            vv = str(v)
                        norm_items.append((k, vv))
                    k2 = tuple(sorted(norm_items))
                    progs.append(f2p.get(k2))

                if any(p is None for p in progs):
                    # Can't resolve all tiles cleanly → skip
                    continue

                n_tiles = len(progs)
                odd0 = oddball_zero_based(odd1)
                if not (0 <= odd0 < n_tiles):
                    continue

                odd_prog = progs[odd0]
                dists: List[float] = []
                for j, ref_prog in enumerate(progs):
                    if j == odd0: continue
                    try:
                        dists.append(float(dist(odd_prog, ref_prog)))
                    except Exception:
                        # robust to rare parser hiccups
                        dists.append(float("nan"))

                valid = [d for d in dists if d == d]  # drop NaNs
                if not valid:
                    continue
                avg_d = sum(valid) / len(valid)

                w.writerow([
                    b.mode, b.domain, b.dim, b.var, t_idx, odd1, n_tiles, f"{avg_d:.6f}", json.dumps(dists)
                ])

                if limit and _ >= limit - 1:
                    break

# ------------------------------ CLI ------------------------------

def main():
    ap = argparse.ArgumentParser(description="Per-trial oddball edit distance (avg over oddball vs each reference).")
    ap.add_argument("--distance", choices=["token", "tree"], default="token",
                    help="Distance type: token-level Levenshtein (default) or tree (requires zss + stitch_core).")
    ap.add_argument("--domains", nargs="*", default=["regular_polygons","glyphs","nuts_and_bolts","totems"])
    ap.add_argument("--roots", nargs="*", default=[str(BASE / "balanced_data"), str(BASE / "data")],
                    help="Roots to scan (balanced_data and/or data).")
    ap.add_argument("--out", default=str(BASE / "analysis" / "per_trial_edit_distance.csv"))
    ap.add_argument("--limit", type=int, default=None, help="Optional max trials per bucket (debug).")
    args = ap.parse_args()

    roots = [Path(r) for r in args.roots]
    buckets = find_buckets(args.domains, roots)
    if not buckets:
        print("No buckets found under:", args.roots, file=sys.stderr)
        sys.exit(2)

    compute_per_trial(buckets, args.distance, Path(args.out), args.limit)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()