"""Parse the T1 snapshot to parquet."""
import gzip, csv, sys
import pandas as pd
import paths

csv.field_size_limit(10**7)
SRC = paths.DATA / 't1_20260811.csv.gz'
DST = paths.WORK / 't1_nyc_state.parquet'

KEEP = ['Service Line Locality', 'Street Address', 'Zip Code',
        'Lead Gooseneck, Pigtail or Connector Currently Present',
        'Current Public Side SL Material', 'Was Public SL Material Ever Previously Lead',
        'Public SL Material Verification Method', 'Public SL Installation or Replacement Date',
        'Public SL Size', 'Customer SL Material', 'Customer SL Material Verification Method',
        'Lead Solder Present', 'Building Type', 'POU or POE Treatment Present',
        'Customer SL Installation or Replacement Date', 'Customer SL Size',
        'SL Category', 'Location']

rows = []
with gzip.open(SRC, 'rt', encoding='utf-8', errors='replace', newline='') as f:
    for i, row in enumerate(csv.DictReader(f)):
        rows.append(tuple(row[c] for c in KEEP))
        if (i + 1) % 1_000_000 == 0:
            print(f'  {i+1:,}', file=sys.stderr, flush=True)

df = pd.DataFrame(rows, columns=KEEP)
print(f'rows parsed: {len(df):,}  (API count at capture: 4,618,115)')
df.to_parquet(DST, index=False)
print(f'wrote {DST}')
