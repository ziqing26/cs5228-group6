#!/usr/bin/env python3
"""Execute new notebooks 06–10, preserving notebooks 01–05."""
from pathlib import Path
import argparse
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--start', type=int, default=6)
parser.add_argument('--end', type=int, default=10)
args = parser.parse_args()
if not 6 <= args.start <= args.end <= 10:
    parser.error('Only notebooks 06–10 may be executed by this runner')
for number in range(args.start, args.end + 1):
    paths = list((ROOT / 'notebooks').glob(f'{number:02d}_*.ipynb'))
    if len(paths) != 1:
        raise RuntimeError(f'Expected exactly one notebook numbered {number}')
    path = paths[0]
    print(f'Executing {path.name}', flush=True)
    nb = nbformat.read(path, as_version=4)
    try:
        NotebookClient(nb, timeout=None, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}}).execute()
    finally:
        nbformat.write(nb, path)
    print(f'Completed {path.name}', flush=True)
