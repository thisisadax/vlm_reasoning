import re
from pathlib import Path
from typing import Dict, Set, Tuple, List


BASE = Path('/scratch/gpfs/nb0564/vlm_reasoning')
DATA_DIR = BASE / 'data' / 'regular_polygons_corrected'
OUT_DIR = BASE / 'output' / 'None' / 'gemini-flash_corrected'


def expected_grid() -> List[Tuple[int, int]]:
    # dim 1: var 0..1; dim 2: var 0..3; dim 3: var 0..5
    exp: List[Tuple[int, int]] = []
    exp += [(1, v) for v in range(0, 2)]
    exp += [(2, v) for v in range(0, 4)]
    exp += [(3, v) for v in range(0, 6)]
    return exp


def find_present_from_outputs() -> Set[Tuple[int, int]]:
    present: Set[Tuple[int, int]] = set()
    if not OUT_DIR.exists():
        return present
    pat = re.compile(r'^regular_polygons_(\d)dim_var(\d+)_dim(\d+)\.csv$')
    for p in OUT_DIR.glob('*.csv'):
        m = pat.search(p.name)
        if not m:
            continue
        lvl = int(m.group(1))  # 1/2/3 only
        var = int(m.group(2))
        present.add((lvl, var))
    return present


def find_present_from_data() -> Set[Tuple[int, int]]:
    present: Set[Tuple[int, int]] = set()
    if not DATA_DIR.exists():
        return present
    for dim_dir in DATA_DIR.glob('dim*'):
        m1 = re.search(r'dim(\d+)$', dim_dir.name)
        if not m1:
            continue
        lvl = int(m1.group(1))
        for var_dir in dim_dir.glob('var*'):
            m2 = re.search(r'var(\d+)$', var_dir.name)
            if not m2:
                continue
            var = int(m2.group(1))
            present.add((lvl, var))
    return present


def main() -> None:
    exp = set(expected_grid())
    out_present = find_present_from_outputs()
    data_present = find_present_from_data()
    both_present = out_present & data_present
    only_data = data_present - out_present
    only_out = out_present - data_present
    missing = exp - (out_present | data_present)

    print('=== regular_polygons (corrected) AUDIT ===')
    print('Expected cells (dim,var):', sorted(exp))
    print('Present in OUTPUT CSVs:', sorted(out_present))
    print('Present in DATA folders:', sorted(data_present))
    print('Present in BOTH (good):', sorted(both_present))
    if only_data:
        print('Data-only (need inference):', sorted(only_data))
    if only_out:
        print('Output-only (unexpected, missing data folder):', sorted(only_out))
    if missing:
        print('Missing entirely (need generate + inference):', sorted(missing))
    else:
        print('No missing cells. ✅')

    # Print ready-to-run tags for any missing or data-only cells
    need_infer = sorted(only_data | missing)
    if need_infer:
        print('\nCommands to generate & infer the following cells:')
        for dim, var in need_infer:
            tag = f'regular_polygons_{dim}dim_var{var}_dim{dim*2}'
            outdir = DATA_DIR / f'dim{dim}' / f'var{var}'
            print(f"- dim={dim} var={var} tag={tag} outdir={outdir}")


if __name__ == '__main__':
    main()



