"""Baseline classifier: CV sensitivity, calibration, transportability."""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

import nyc_core as C
import paths

PHYS = ['EXCAVATION', 'FIELD INSPECTION']
MODEL_METH = 'STATISTICAL ANALYSIS/PREDICTIVE MODEL'
LAT0 = 40.7
M_LAT, M_LON = 110_540, 111_320 * np.cos(np.radians(LAT0))
SEED = 20260818


def hgb(seed=0):
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                          max_depth=6, random_state=seed)


def blocks(lat, lon, metres):
    """Grid-cell id at a given edge length -- the CV grouping variable."""
    return (np.floor(lat * M_LAT / metres).astype(np.int64) * 100_000 +
            np.floor(lon * M_LON / metres).astype(np.int64))


def featurise(d, btypes, zips, with_zip):
    cols = [d['lat'].values, d['lon'].values,
            d['btype'].map({b: i for i, b in enumerate(btypes)}).fillna(-1).values]
    cat = [False, False, True]
    if with_zip:
        cols.append(d['zip5'].map({z: i for i, z in enumerate(zips)}).fillna(-1).values)
        cat.append(True)
    return np.column_stack(cols).astype(float), cat


def cv_auc(X, y, groups, cat):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan)
    for fold, (tr, te) in enumerate(cv.split(X, y, groups)):
        m = hgb(fold); m.set_params(categorical_features=cat)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    ok = ~np.isnan(oof)
    return oof, dict(auc=roc_auc_score(y[ok], oof[ok]),
                     ap=average_precision_score(y[ok], oof[ok]),
                     brier=brier_score_loss(y[ok], oof[ok]), n=int(ok.sum()))


def main():
    pub, meta = C.load_nyc(apply_bbox=True)
    print('input:', {k: f'{v:,}' for k, v in meta.items()})

    phys = pub[pub['pubmeth'].isin(PHYS)].copy()
    n_phys_all = len(phys)
    amb = ~phys['pubmat'].isin([C.LEAD_MAT, 'Known Other'])
    print(f'\nphysically-verified addresses: {n_phys_all:,}')
    print(f'  dropped as neither clean Lead nor clean Non-Lead: {int(amb.sum()):,} '
          f'({amb.mean():.2%})  ->  {phys.loc[amb, "pubmat"].value_counts().to_dict()}')
    phys = phys[~amb].copy()
    y = phys['is_lead'].values.astype(int)
    print(f'  training population: {len(y):,}   Lead {y.sum():,} ({y.mean():.2%})')

    cleared = pub[pub['pubmeth'] == MODEL_METH].copy()
    print(f'model-cleared addresses to score: {len(cleared):,} '
          f'(recorded material: {cleared["pubmat"].unique().tolist()})')

    btypes = sorted(set(phys['btype']) | set(cleared['btype']))
    zips = sorted(set(phys['zip5']) | set(cleared['zip5']))
    results = {'input': meta, 'training_n': int(len(y)), 'training_lead_rate': float(y.mean()),
               'cleared_n': int(len(cleared))}

    # CV grouping sensitivity
    print('\n=== 1. CV design sensitivity: AUC vs. how far apart train and test are ===')
    X_nz, cat_nz = featurise(phys, btypes, zips, with_zip=False)
    X_z, cat_z = featurise(phys, btypes, zips, with_zip=True)
    grouping = {'ZIP (session 2)': phys['zip5'].values}
    for m in (1000, 2000, 5000, 10000):
        grouping[f'{m//1000} km block'] = blocks(phys['lat'].values, phys['lon'].values, m)
    rows, oof_ref = [], None
    for name, g in grouping.items():
        oof, s = cv_auc(X_nz, y, g, cat_nz)
        _, sz = cv_auc(X_z, y, g, cat_z)
        rows.append(dict(grouping=name, n_groups=len(np.unique(g)),
                         auc_no_zip=s['auc'], auc_with_zip=sz['auc'],
                         ap=s['ap'], brier=s['brier']))
        if name.startswith('2 km'):
            oof_ref = oof
        print(f'  {name:>16}  groups={len(np.unique(g)):>6,}  '
              f'AUC(no ZIP)={s["auc"]:.3f}  AUC(with ZIP)={sz["auc"]:.3f}  '
              f'AP={s["ap"]:.3f}  Brier={s["brier"]:.4f}')
    results['cv_sensitivity'] = rows

    # calibration
    print('\n=== 2. Calibration of the 2 km-block out-of-fold predictions ===')
    q = pd.qcut(oof_ref, 10, labels=False, duplicates='drop')
    cal = pd.DataFrame({'p': oof_ref, 'y': y, 'q': q}).groupby('q').agg(
        n=('y', 'size'), mean_p=('p', 'mean'), obs=('y', 'mean')).reset_index()
    cal['ratio'] = cal['mean_p'] / cal['obs'].replace(0, np.nan)
    print(cal.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    overall = oof_ref.sum() / y.sum()
    lo = oof_ref < 0.10
    print(f'\n  sum(p)/sum(y) over the whole training set : {overall:.4f}')
    print(f'  sum(p)/sum(y) restricted to p<0.10        : {oof_ref[lo].sum()/max(y[lo].sum(),1):.4f} '
          f'(n={lo.sum():,}) -- this is the band the model-cleared population sits in')
    results['calibration'] = dict(deciles=cal.to_dict('records'),
                                  sum_ratio_all=float(overall),
                                  sum_ratio_low_band=float(oof_ref[lo].sum() / max(y[lo].sum(), 1)))

    # transportability
    print('\n=== 3. Transportability: can a classifier tell the two populations apart? ===')
    both = pd.concat([phys.assign(t=0), cleared.assign(t=1)])
    Xb, catb = featurise(both, btypes, zips, with_zip=False)
    gb = blocks(both['lat'].values, both['lon'].values, 2000)
    _, s = cv_auc(Xb, both['t'].values, gb, catb)
    print(f'  AUC(model-cleared vs. physically-verified) = {s["auc"]:.3f}   '
          f'(0.5 = same population, 1.0 = disjoint)')
    results['transportability_auc'] = s['auc']

    # score the cleared population
    print('\n=== 4. Expected lead lines among the model-cleared addresses ===')
    scored = {}
    for label, with_zip in (('no ZIP (headline)', False), ('with ZIP', True)):
        Xtr, cat = featurise(phys, btypes, zips, with_zip)
        Xte, _ = featurise(cleared, btypes, zips, with_zip)
        m = hgb(0); m.set_params(categorical_features=cat)
        m.fit(Xtr, y)
        p = m.predict_proba(Xte)[:, 1]
        per_b = {b: dict(n=int((cleared['loc'] == b).sum()),
                         mean_p=float(p[(cleared['loc'] == b).values].mean()),
                         implied=float(p[(cleared['loc'] == b).values].sum()))
                 for b in C.BOROUGHS}
        scored[label] = dict(mean_p=float(p.mean()), implied=float(p.sum()), by_borough=per_b)
        print(f'\n  -- {label} --   mean P(Lead)={p.mean():.3%}   implied={p.sum():.0f}')
        for b in C.BOROUGHS:
            d = per_b[b]
            print(f'     {C.BOROUGH_NAME[b]:<14} n={d["n"]:>6,}  mean P={d["mean_p"]:.3%}  '
                  f'implied={d["implied"]:.0f}')
    results['scored'] = scored

    with open(paths.WORK / 'baseline2_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print('\nwrote work/baseline2_results.json')


if __name__ == '__main__':
    main()
