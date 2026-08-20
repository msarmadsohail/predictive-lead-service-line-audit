"""Join service-line addresses to the nearest MapPLUTO lot."""
import gzip, json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import nyc_core as C
import paths

CAP_M = 100.0
LAT0 = 40.7
M_LAT, M_LON = 110_540, 111_320 * np.cos(np.radians(LAT0))
OUT = paths.WORK / 'nyc_joined.parquet'


def main():
    with gzip.open(paths.WORK / 'mappluto.json.gz', 'rt') as f:
        pl = pd.DataFrame(json.load(f))
    print(f'MapPLUTO lots: {len(pl):,}')
    for c in ['latitude', 'longitude', 'yearbuilt', 'yearalter1', 'unitsres',
              'unitstotal', 'numfloors', 'bldgarea', 'lotarea', 'numbldgs']:
        pl[c] = pd.to_numeric(pl.get(c), errors='coerce')
    pl = pl.dropna(subset=['latitude', 'longitude'])
    # yearbuilt 0 = unknown sentinel
    pl['yearbuilt'] = pl['yearbuilt'].where(pl['yearbuilt'].between(1800, 2026))
    print(f'  with usable centroid: {len(pl):,}   with usable yearbuilt: '
          f'{pl["yearbuilt"].notna().sum():,} ({pl["yearbuilt"].notna().mean():.1%})')

    pub, meta = C.load_nyc(apply_bbox=True)
    print(f'LSL addresses: {len(pub):,}')

    lot_xy = np.column_stack([pl['longitude'].values * M_LON, pl['latitude'].values * M_LAT])
    tree = cKDTree(lot_xy)
    sl_xy = np.column_stack([pub['lon'].values * M_LON, pub['lat'].values * M_LAT])
    dist, idx = tree.query(sl_xy, k=1)

    print(f'\nmatch distance to nearest lot centroid (m): '
          f'median {np.median(dist):.1f}  p90 {np.percentile(dist,90):.1f}  '
          f'p99 {np.percentile(dist,99):.1f}')
    ok = dist <= CAP_M
    print(f'  within {CAP_M:.0f} m: {ok.sum():,} ({ok.mean():.2%})   '
          f'beyond, dropped from the joined feature set: {(~ok).sum():,}')

    take = pl.iloc[idx].reset_index(drop=True)
    for c in ['yearbuilt', 'yearalter1', 'unitsres', 'unitstotal', 'numfloors',
              'bldgarea', 'lotarea', 'numbldgs', 'bldgclass', 'landuse']:
        v = take[c].values
        pub['pluto_' + c] = np.where(ok, v, np.nan) if take[c].dtype != object else \
            np.where(ok, v, None)
    pub['pluto_dist_m'] = dist
    pub['pluto_matched'] = ok

    yb = pub.loc[pub['pluto_matched'], 'pluto_yearbuilt']
    print(f'  of matched, with a usable yearbuilt: {yb.notna().sum():,} ({yb.notna().mean():.1%})')

    print('\n=== Does building age actually separate lead from non-lead here? ===')
    phys = pub[pub['pubmeth'].isin(['EXCAVATION', 'FIELD INSPECTION']) &
               pub['pubmat'].isin([C.LEAD_MAT, 'Known Other']) &
               pub['pluto_yearbuilt'].notna()].copy()
    bins = [1800, 1900, 1920, 1940, 1950, 1960, 1980, 2000, 2027]
    phys['era'] = pd.cut(phys['pluto_yearbuilt'], bins, right=False)
    t = phys.groupby('era', observed=True)['is_lead'].agg(n='size', lead='sum')
    t['lead_pct'] = 100 * t['lead'] / t['n']
    print(t.to_string())

    print('\n=== Era mix: physically verified vs. model-cleared ===')
    cl = pub[(pub['pubmeth'] == 'STATISTICAL ANALYSIS/PREDICTIVE MODEL') &
             pub['pluto_yearbuilt'].notna()].copy()
    cl['era'] = pd.cut(cl['pluto_yearbuilt'], bins, right=False)
    mix = pd.DataFrame({'physical %': 100 * phys['era'].value_counts(normalize=True),
                        'model-cleared %': 100 * cl['era'].value_counts(normalize=True)})
    print(mix.sort_index().round(2).to_string())
    print(f"\nmedian year built -- physically verified: {phys['pluto_yearbuilt'].median():.0f}   "
          f"model-cleared: {cl['pluto_yearbuilt'].median():.0f}")

    pub.to_parquet(OUT, index=False)
    print(f'\nwrote {OUT}  ({len(pub):,} rows)')


if __name__ == '__main__':
    main()
