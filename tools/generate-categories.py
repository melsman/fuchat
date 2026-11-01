#!/usr/bin/env python3
"""Generate Unicode letter/number ranges from pinned Unicode 17.0.0 data."""
import hashlib
from pathlib import Path
import sys
raw = Path(sys.argv[1]).read_bytes()
if hashlib.sha256(raw).hexdigest() != '2e1efc1dcb59c575eedf5ccae60f95229f706ee6d031835247d843c11d96470c':
    raise SystemExit('Expected Unicode 17.0.0 UnicodeData.txt')
ranges = []
first = None
for line in raw.decode().splitlines():
    f = line.split(';')
    cp = int(f[0], 16)
    kind = 1 if f[2].startswith('L') else 2 if f[2].startswith('N') else 0
    if f[1].endswith(', First>'):
        first = cp
        continue
    lo = first if f[1].endswith(', Last>') else cp
    first = None
    if kind:
        if ranges and ranges[-1][1] + 1 == lo and ranges[-1][2] == kind:
            ranges[-1] = (ranges[-1][0], cp, kind)
        else:
            ranges.append((lo, cp, kind))
print('(* Generated Unicode 17.0.0 letter (1) and number (2) categories.')
print(' * Source: https://www.unicode.org/Public/17.0.0/ucd/UnicodeData.txt')
print(' * Regenerate using tools/generate-categories.py; license: unicode-license.txt. *)')
print('structure UnicodeCategories = struct\n  val ranges = Vector.fromList [')
print(',\n'.join(f'    ({lo}, {hi}, {kind})' for lo, hi, kind in ranges))
print('  ]\nend')
