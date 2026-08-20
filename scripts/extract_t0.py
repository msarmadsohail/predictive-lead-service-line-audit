"""Parse the T0 snapshot to parquet."""
import gzip, csv, sys
import pandas as pd
import paths

csv.field_size_limit(10**7)
SRC = paths.DATA / 't0_20250622.csv.gz'
DST = paths.WORK / 't0_state.parquet'

WANT = ['Service Line Locality', 'Street Address', 'Zip Code',
        'Current Public Side SL Material', 'Public SL Material Verification Method',
        'Customer SL Material', 'Customer SL Material Verification Method',
        'Building Type', 'SL Category', 'Location']

rows = []
with gzip.open(SRC, 'rt', encoding='utf-8', errors='replace', newline='') as f:
    r = csv.DictReader(f)
    have = [c for c in WANT if c in r.fieldnames]
    missing = [c for c in WANT if c not in r.fieldnames]
    print(f'T0 columns: {len(r.fieldnames)}   using {len(have)}   missing {missing}')
    for i, row in enumerate(r):
        rows.append(tuple(row[c] for c in have))
        if (i + 1) % 1_000_000 == 0:
            print(f'  {i+1:,}', file=sys.stderr, flush=True)

df = pd.DataFrame(rows, columns=have)
print(f'rows parsed: {len(df):,}  (API count at capture: 3,747,025)')
assert len(df) == 3_747_025, 'row count does not match the recorded API count -- stop'
df.to_parquet(DST, index=False)
print(f'wrote {DST}')
