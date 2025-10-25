import re
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd


BASE = Path("/scratch/gpfs/nb0564/vlm_reasoning")
BAL_DATA = BASE / "balanced_data"
BAL_OUT = BASE / "balanced_output"

VAR_MAX = {1: 1, 2: 3, 3: 5}

# domain, filename regex, output subdir under balanced_output/*, data domain subdir under balanced_data/*
DOMAINS = [
    ("regular_polygons", r"^regular_polygons_(\d)dim_var(\d+)_dim\d+\.csv$", "None/gemini-flash", "regular_polygons"),
    ("glyphs", r"^glyphs_dim(\d)_var(\d+)_dim\d+\.csv$", "glyphs/gemini-flash", "glyphs"),
    ("nuts_and_bolts", r"^nab_dim(\d)_var(\d+)_dim\d+\.csv$", "nuts_and_bolts/gemini-flash", "nuts_and_bolts"),
    ("totems", r"^totems_(\d)mod_var(\d+)_dim\d+\.csv$", "totems/gemini-flash", "totems"),
]


def get_meta(trial: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(trial, dict):
        m = trial.get("meta")
        return m if isinstance(m, dict) else {}
    return {}


def get_tiles(trial: Dict[str, Any]):
    return trial.get("tiles") if isinstance(trial, dict) else None


def get_feats(tile: Any) -> Dict[str, Any]:
    if not isinstance(tile, dict):
        return {}
    f = tile.get("features")
    return f if isinstance(f, dict) else {}


def split_refs_odd(trial: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return a list of reference tiles and the oddball tile (if found)."""
    tiles = get_tiles(trial)
    meta = get_meta(trial)
    odd_idx = meta.get("oddball_idx") if isinstance(meta.get("oddball_idx"), int) else trial.get("oddball_idx")

    # common dict encoding
    if isinstance(tiles, dict):
        refs = tiles.get("refs") or tiles.get("ref_tiles") or tiles.get("references")
        odd = tiles.get("odd") or tiles.get("oddball")
        if isinstance(refs, list) and refs:
            return refs, odd if isinstance(odd, dict) else None
        arr = tiles.get("all")
        if isinstance(arr, list) and isinstance(odd_idx, int) and 0 <= odd_idx < len(arr):
            odd = arr[odd_idx]
            refs = [arr[i] for i in range(len(arr)) if i != odd_idx]
            return refs, odd

    # list encoding
    if isinstance(tiles, list) and isinstance(odd_idx, int) and 0 <= odd_idx < len(tiles):
        odd = tiles[odd_idx]
        refs = [tiles[i] for i in range(len(tiles)) if i != odd_idx]
        return refs, odd

    # top-level fallbacks
    refs = trial.get("ref_tiles") if isinstance(trial.get("ref_tiles"), list) else []
    odd = trial.get("oddball") if isinstance(trial.get("oddball"), dict) else None
    return refs, odd


def audit_var_balanced() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    problems: List[Tuple[str, str, str, str]] = []
    summary: List[Tuple[str, int, int, int, str, str]] = []
    coverage: List[Tuple[str, int, List[int], List[int], bool]] = []

    for domain, pat_str, out_rel, data_dom in DOMAINS:
        out_dir = BAL_OUT / "var_balanced" / out_rel
        data_root = BAL_DATA / "var_balanced" / data_dom
        if not out_dir.exists():
            continue
        rx = re.compile(pat_str)
        seen = set()
        for f in sorted(out_dir.glob("*.csv")):
            m = rx.match(f.name)
            if not m:
                problems.append((domain, str(f), "filename_parse", "cannot_parse_dim_var"))
                continue
            dim = int(m.group(1))
            var = int(m.group(2))
            seen.add((var, dim))
            tjson = data_root / f"dim{dim}" / f"var{var}" / "trials.jsonl"

            # read CSV
            try:
                df = pd.read_csv(f)
            except Exception as e:
                problems.append((domain, str(f), "csv_read_error", str(e)))
                continue
            n_csv = len(df)

            # read trials
            trials = []
            if tjson.exists():
                with open(tjson, "r") as fh:
                    for line in fh:
                        try:
                            trials.append(json.loads(line))
                        except Exception:
                            pass
            else:
                problems.append((domain, str(f), "missing_trials", str(tjson)))
            n_trials = len(trials)
            if n_trials and n_csv != n_trials:
                problems.append((domain, str(f), "row_count_mismatch", f"csv={n_csv} trials={n_trials}"))

            # label + oddball checks (sample up to 50)
            bad_dim = bad_var = bad_odd_ref = bad_odd_match = 0
            for t in trials[:50]:
                meta = get_meta(t)
                # Fall back to top-level fields if meta is missing these
                dim_val = meta.get("n_dimensions") or t.get("n_dimensions")
                var_val = meta.get("reference_variance") or t.get("reference_variance")
                if dim_val != dim:
                    bad_dim += 1
                if var_val != var:
                    bad_var += 1
                odd_abs = t.get("oddball_abstraction") or meta.get("oddball_abstraction")
                refs, odd = split_refs_odd(t)
                if odd_abs and refs and isinstance(odd, dict):
                    ref_vals = {get_feats(r).get(odd_abs) for r in refs if isinstance(r, dict)}
                    if len(ref_vals) != 1:
                        bad_odd_ref += 1
                    ref_val = next(iter(ref_vals)) if ref_vals else None
                    odd_val = get_feats(odd).get(odd_abs, ref_val)
                    if odd_val == ref_val:
                        bad_odd_match += 1
            status = (
                "ok"
                if not (bad_dim or bad_var or bad_odd_ref or bad_odd_match)
                else f"issues dim={bad_dim} var={bad_var} odd_ref={bad_odd_ref} odd_match={bad_odd_match}"
            )
            summary.append((domain, dim, var, n_csv, status, f.name))

        # coverage per var
        for var in range(0, 6):
            dims_present = sorted([d for (v, d) in seen if v == var])
            dims_expected = sorted([d for d in (1, 2, 3) if VAR_MAX[d] >= var])
            coverage.append((domain, var, dims_present, dims_expected, dims_present == dims_expected))

    S = (
        pd.DataFrame(summary, columns=["domain", "dim", "var", "n", "status", "file"])
        if summary
        else pd.DataFrame(columns=["domain", "dim", "var", "n", "status", "file"])
    )
    P = (
        pd.DataFrame(problems, columns=["domain", "file", "issue", "detail"])
        if problems
        else pd.DataFrame(columns=["domain", "file", "issue", "detail"])
    )
    C = (
        pd.DataFrame(coverage, columns=["domain", "var", "dims_present", "dims_expected", "exact_match"])
        if coverage
        else pd.DataFrame(columns=["domain", "var", "dims_present", "dims_expected", "exact_match"])
    )
    return S, P, C


def dim_balanced_accuracy() -> pd.DataFrame:
    rows: List[Tuple[str, int, int, float]] = []
    for domain, pat_str, out_rel, _ in DOMAINS:
        out_dir = BAL_OUT / "dim_balanced" / out_rel
        if not out_dir.exists():
            continue
        rx = re.compile(pat_str)
        for f in sorted(out_dir.glob("*.csv")):
            m = rx.match(f.name)
            if not m:
                continue
            dim = int(m.group(1))
            var = int(m.group(2))
            try:
                df = pd.read_csv(f)
            except Exception:
                continue
            # Prefer computing correctness as (response == answer)
            resp_col = None
            for c in ("response", "model_response", "prediction", "pred", "model_answer"):
                if c in df.columns:
                    resp_col = c
                    break
            if resp_col and "answer" in df.columns:
                df["is_correct"] = (
                    df[resp_col].astype(str).str.strip() == df["answer"].astype(str).str.strip()
                ).astype(int)
            elif "is_correct" in df.columns:
                # Fallback to precomputed field if present
                df["is_correct"] = df["is_correct"].astype(int)
            else:
                continue
            rows.append((domain, dim, var, float(df["is_correct"].mean())))
    A = (
        pd.DataFrame(rows, columns=["domain", "dim", "var", "acc"])
        if rows
        else pd.DataFrame(columns=["domain", "dim", "var", "acc"])
    )
    return A


def main() -> None:
    S, P, C = audit_var_balanced()
    print("=== VAR-BALANCED audit ===")
    if not S.empty:
        print("Counts by domain,var,dim (first 24):")
        print(S.groupby(["domain", "var", "dim"])['n'].sum().head(24))
        bad = S[S["status"] != "ok"]
        print("\nStatuses OK?", bad.empty)
        if not bad.empty:
            print(bad.head(20).to_string(index=False))
    else:
        print("NO FILES")
    print("\nDim coverage per var (first 20):")
    if not C.empty:
        print(C.head(20).to_string(index=False))
    else:
        print("NONE")
    print("\nProblems (first 30):")
    if not P.empty:
        print(P.head(30).to_string(index=False))
    else:
        print("NONE")

    print("\n=== DIM-BALANCED accuracy by domain,dim ===")
    A = dim_balanced_accuracy()
    if A.empty:
        print("NO DATA")
    else:
        M = A.groupby(["domain", "dim"])['acc'].mean().unstack(fill_value=float('nan'))
        print(M)
        print("\nOverall by dim:")
        print(A.groupby('dim')['acc'].mean())


if __name__ == "__main__":
    main()



