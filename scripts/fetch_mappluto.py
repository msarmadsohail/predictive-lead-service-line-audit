"""Download MapPLUTO tax lots from NYC Open Data."""
import json, sys, time, urllib.request, urllib.parse, gzip
import paths

BASE = 'https://data.cityofnewyork.us/resource/64uk-42ks.json'
COLS = ['bbl','borough','latitude','longitude','yearbuilt','yearalter1','unitsres',
        'unitstotal','numfloors','bldgclass','landuse','bldgarea','lotarea','zipcode','numbldgs']
PAGE = 50_000
OUT = paths.WORK / 'mappluto.json.gz'


def get(url, tries=5):
    for t in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            print(f'  retry {t+1}/{tries} after {e}', file=sys.stderr, flush=True)
            time.sleep(5 * (t + 1))
    raise RuntimeError(f'failed: {url}')


# canary
canary = get(BASE + '?' + urllib.parse.urlencode({'$select': 'count(bbl)'}))
total = int(canary[0]['count_bbl'])
print(f'MapPLUTO total lots reported by API: {total:,}', flush=True)

rows, off = [], 0
while True:
    q = urllib.parse.urlencode({'$select': ','.join(COLS), '$limit': PAGE,
                                '$offset': off, '$order': 'bbl'})
    page = get(f'{BASE}?{q}')
    rows.extend(page)
    print(f'  offset {off:,} -> +{len(page):,} (running {len(rows):,})', flush=True)
    if len(page) < PAGE:
        break
    off += PAGE

print(f'fetched {len(rows):,} of {total:,} ({len(rows)/total:.2%})')
assert len(rows) >= total * 0.99, f'SHORT READ: {len(rows):,} < {total:,} -- do not use'
with gzip.open(OUT, 'wt') as f:
    json.dump(rows, f)
print(f'wrote {OUT}')
