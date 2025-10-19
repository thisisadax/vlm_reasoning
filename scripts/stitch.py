#!/usr/bin/env python3
import os, io, csv, glob, argparse, hashlib
from typing import List, Dict, Any, Iterable, Optional

import stitch_core  # your module

def tokenize_program(p: str) -> List[str]:
    s = (p or "").replace("(", " ").replace(")", " ")
    for ch in ["\t", "\r", "\n"]:
        s = s.replace(ch, " ")
    return [t for t in s.split() if t]

def count_tokens(p: str) -> int:
    return len(tokenize_program(p))

def build_minimal_dsl(programs: List[str]) -> Dict[str, Any]:
    toks = []
    for p in programs: toks.extend(tokenize_program(p))
    uniq = list(dict.fromkeys(toks))
    return {"logVariable": 0.0,
            "productions": [{"expression": t, "logProbability": 0.0} for t in uniq]}

def programs_to_frontiers(programs: List[str]) -> List[Dict[str, Any]]:
    return [{"task":{"name":f"task_{i}"},
             "programs":[{"program":p, "logProbability":0.0}]} for i,p in enumerate(programs)]

def _csv_reader(path: str) -> Iterable[Dict[str, str]]:
    with open(path, "r", newline="") as f:
        raw = f.read()
    if raw.startswith("\ufeff"): raw = raw.lstrip("\ufeff")
    buf = io.StringIO(raw)
    for r in csv.DictReader(buf): yield r

CAND_IDS = ["stable_id","id","image_id","program_id"]
def _stable_id(row: Dict[str,str], prog: str) -> str:
    for k in CAND_IDS:
        v = row.get(k)
        if v not in (None, ""): return str(v)
    return hashlib.sha1((prog or "").encode("utf-8")).hexdigest()

def collect_for_task(task_root: str, task_name: str):
    corpus = []  # rows of (source_csv, rel_id, program_string, stable_id)
    for meta in sorted(glob.glob(os.path.join(task_root, "**", "metadata.csv"), recursive=True)):
        ridx = 0
        for r in _csv_reader(meta):
            prog = r.get("program_string", "")
            if not prog: continue
            corpus.append((os.path.abspath(meta), ridx, prog, _stable_id(r, prog)))
            ridx += 1
    return corpus

def run_stitch(programs: List[str], iterations: int, max_arity: int, threads: int, silent: bool) -> List[str]:
    dc_json = {"DSL": build_minimal_dsl(programs),
               "frontiers": programs_to_frontiers(programs)}
    sa = stitch_core.from_dreamcoder(dc_json)
    comp = stitch_core.compress(
        programs=sa["programs"], iterations=iterations, max_arity=max_arity,
        threads=threads, silent=silent, tasks=[t["name"] for t in sa["tasks"]],
        name_mapping=sa["name_mapping"], rewritten_dreamcoder=True,
    )
    out = [str(p) for p in getattr(comp, "rewritten", [])]
    if len(out) < len(programs): out.extend(programs[len(out):])
    return out

def write_csv(path: str, rows: List[Dict[str,Any]]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fn = ["task","source_csv","rel_id","stable_id","mdl","compression_ratio","compressed_program","original_program"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fn); w.writeheader()
        for r in rows: w.writerow({k: r.get(k, "") for k in fn})
    print(f"Wrote {path} ({len(rows)} rows)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--out_dir", default=None, help="Dir for outputs; defaults to data_dir")
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--max_arity", type=int, default=3)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--silent", action="store_true")
    args = ap.parse_args()

    out_dir = args.out_dir or args.data_dir
    merged_rows = []

    for task in args.tasks:
        root = os.path.join(args.data_dir, task)
        corpus = collect_for_task(root, task)
        if not corpus:
            print(f"[{task}] no programs found"); continue
        programs = [p for (_,_,p,_) in corpus]
        rewritten = run_stitch(programs, args.iterations, args.max_arity, args.threads, args.silent)

        rows = []
        for (source_csv, rel_id, orig, sid), comp_prog in zip(corpus, rewritten):
            dl  = max(1, count_tokens(orig))
            mdl = max(1, count_tokens(comp_prog))
            rows.append({
                "task": task, "source_csv": source_csv, "rel_id": rel_id, "stable_id": sid,
                "mdl": mdl, "compression_ratio": dl/mdl,
                "compressed_program": comp_prog, "original_program": orig,
            })
        # per-domain file
        per_task_csv = os.path.join(out_dir, f"stitch_results_{task}.csv")
        write_csv(per_task_csv, rows)
        merged_rows.extend(rows)

    if merged_rows:
        write_csv(os.path.join(out_dir, "stitch_results_all.csv"), merged_rows)

if __name__ == "__main__":
    main()