"""Eleven estimators of the implied lead count."""
import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier

import nyc_core as C
import paths

JOINED = paths.WORK / 'nyc_joined.parquet'
MODEL_METH = 'STATISTICAL ANALYSIS/PREDICTIVE MODEL'
PHYS = ['EXCAVATION', 'FIELD INSPECTION']
ERA_BINS = [1800, 1900, 1920, 1940, 1950, 1960, 1980, 2000, 2027]
MIN_CELL = 20
LAT0 = 40.7
M_LAT, M_LON = 110_540, 111_320 * np.cos(np.radians(LAT0))


def load():
    d = pd.read_parquet(JOINED)
    d['era'] = pd.cut(d['pluto_yearbuilt'], ERA_BINS, right=False).astype(str)
    phys = d[d['pubmeth'].isin(PHYS) & d['pubmat'].isin([C.LEAD_MAT, 'Known Other'])].copy()
    cleared = d[d['pubmeth'] == MODEL_METH].copy()
    return phys, cleared


def stratified(phys, cleared, keys, name):
    """Apply each stratum's observed physical lead rate to the model-cleared count in it."""
    g = phys.groupby(keys, observed=True)['is_lead'].agg(n='size', lead='sum')
    g = g[g['n'] >= MIN_CELL]
    rate = (g['lead'] / g['n']).rename('rate')
    m = cleared.join(rate, on=keys)
    scored = m['rate'].notna()
    implied = float(m.loc[scored, 'rate'].sum())
    cov = float(scored.mean())
    return dict(estimator=name, scored=int(scored.sum()), n=len(cleared), coverage=cov,
                implied_scored=implied,
                implied_extrapolated=implied / cov if cov else float('nan'),
                per_address_rate=implied / max(scored.sum(), 1))


def spatial_nn(phys, cleared, caliper, use_era, name):
    tot_imp, tot_scored = 0.0, 0
    strata = ['loc', 'btype'] + (['era'] if use_era else [])
    for k, cl in cleared.groupby(strata, observed=True):
        ph = phys
        for col, v in zip(strata, k if isinstance(k, tuple) else (k,)):
            ph = ph[ph[col] == v]
        if len(ph) < MIN_CELL:
            continue
        tree = cKDTree(np.column_stack([ph['lon'] * M_LON, ph['lat'] * M_LAT]))
        yv = ph['is_lead'].values.astype(float)
        nb = tree.query_ball_point(np.column_stack([cl['lon'] * M_LON, cl['lat'] * M_LAT]),
                                   r=caliper)
        for idxs in nb:
            if len(idxs) < MIN_CELL:
                continue
            tot_imp += yv[idxs].mean(); tot_scored += 1
    cov = tot_scored / len(cleared)
    return dict(estimator=name, scored=tot_scored, n=len(cleared), coverage=cov,
                implied_scored=tot_imp,
                implied_extrapolated=tot_imp / cov if cov else float('nan'),
                per_address_rate=tot_imp / max(tot_scored, 1))


NUM = ['lat', 'lon', 'pluto_yearbuilt', 'pluto_unitsres', 'pluto_numfloors',
       'pluto_bldgarea', 'pluto_lotarea']


def gbm(phys, cleared, use_era, name):
    cols = ['lat', 'lon'] + (NUM[2:] if use_era else [])
    cats = sorted(set(phys['btype']) | set(cleared['btype']))
    cmap = {b: i for i, b in enumerate(cats)}

    def X(d):
        a = [d[c].astype(float).values for c in cols]
        a.append(d['btype'].map(cmap).fillna(-1).values.astype(float))
        return np.column_stack(a)

    cat_mask = [False] * len(cols) + [True]
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=6,
                                       random_state=0, categorical_features=cat_mask)
    m.fit(X(phys), phys['is_lead'].values.astype(int))
    p = m.predict_proba(X(cleared))[:, 1]
    return dict(estimator=name, scored=len(cleared), n=len(cleared), coverage=1.0,
                implied_scored=float(p.sum()), implied_extrapolated=float(p.sum()),
                per_address_rate=float(p.mean())), p


def main():
    phys, cleared = load()
    print(f'physically-verified (clean-label) addresses : {len(phys):,}  '
          f'lead {phys["is_lead"].sum():,} ({phys["is_lead"].mean():.2%})')
    print(f'model-cleared addresses                     : {len(cleared):,}')
    print(f'  with a PLUTO year built                   : {cleared["pluto_yearbuilt"].notna().sum():,} '
          f'({cleared["pluto_yearbuilt"].notna().mean():.1%})')

    rows = []
    rows.append(stratified(phys, cleared, ['loc'], 'A1 borough only'))
    rows.append(stratified(phys, cleared, ['loc', 'btype'], 'A2 borough x building type'))
    rows.append(stratified(phys, cleared, ['loc', 'era'], 'A3 borough x era'))
    rows.append(stratified(phys, cleared, ['loc', 'btype', 'era'], 'A4 borough x type x era'))
    rows.append(stratified(phys, cleared, ['zip5', 'era'], 'A5 ZIP x era'))
    for cal in (500, 1000):
        rows.append(spatial_nn(phys, cleared, cal, False, f'B{cal} spatial {cal} m, no era'))
        rows.append(spatial_nn(phys, cleared, cal, True, f'B{cal}e spatial {cal} m + era'))
    g1, _ = gbm(phys, cleared, False, 'C1 GBM lat/lon/type (no era)')
    g2, p2 = gbm(phys, cleared, True, 'C2 GBM + PLUTO era & built form')
    rows += [g1, g2]

    df = pd.DataFrame(rows)
    pd.set_option('display.width', 200)
    print('\n=== Expected lead public-side lines among the model-cleared population ===')
    print(df.assign(coverage=lambda d: (100 * d.coverage).round(1),
                    per_address_rate=lambda d: (100 * d.per_address_rate).round(2))
            .rename(columns={'coverage': 'cov %', 'per_address_rate': 'rate %'})
            .to_string(index=False, float_format=lambda v: f'{v:,.0f}'))

    ERA_AWARE = {'A3 borough x era', 'A4 borough x type x era', 'A5 ZIP x era',
                 'B500e spatial 500 m + era', 'B1000e spatial 1000 m + era',
                 'C2 GBM + PLUTO era & built form'}
    era_aware = df[df.estimator.isin(ERA_AWARE)]
    print(f'\nera-aware estimators, extrapolated: '
          f'{era_aware.implied_extrapolated.min():,.0f} - {era_aware.implied_extrapolated.max():,.0f}')
    naive = df[~df.estimator.isin(ERA_AWARE)]
    print(f'era-blind estimators, extrapolated : '
          f'{naive.implied_extrapolated.min():,.0f} - {naive.implied_extrapolated.max():,.0f}')

    print('\n=== Borough split, C2 (the era-aware GBM) ===')
    cleared = cleared.assign(p=p2)
    b = cleared.groupby('loc')['p'].agg(n='size', mean_p='mean', implied='sum')
    b['name'] = [C.BOROUGH_NAME[i] for i in b.index]
    print(b.assign(mean_p=lambda d: (100 * d.mean_p).round(2)).to_string())

    with open(paths.WORK / 'estimate_results.json', 'w') as f:
        json.dump(dict(rows=rows, borough_c2=b.reset_index().to_dict('records')), f,
                  indent=2, default=float)
    print('\nwrote work/estimate_results.json')


if __name__ == '__main__':
    main()
