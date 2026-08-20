"""Per-system output variance for model-classified addresses."""
import json, re
import numpy as np
import pandas as pd
import paths

SRC = paths.WORK / 't1_nyc_state.parquet'
OUT = paths.WORK / 'statewide_screen.json'
MIN_N = 100

MODEL_RE = re.compile(r'PREDICTIVE MODEL|STATISTICAL ANALYSIS')
MODEL_STRICT = 'STATISTICAL ANALYSIS/PREDICTIVE MODEL'
PHYS_RE = re.compile(r'EXCAVAT|FIELD INSPECT|FIELD INVESTIG|FIELD INSP')


def norm(s):
    return s.str.strip().str.upper().str.replace(r'\s+', ' ', regex=True)


def classify_material(v):
    """Six outcome classes. Order matters: the lead tests run before the unknown tests."""
    t = v.strip().upper()
    if not t:
        return 'blank'
    if 'COULD BE LEAD' in t:
        return 'unknown_could_be_lead'
    if 'UNLIKELY LEAD' in t:
        return 'unknown_unlikely_lead'
    if 'LEAD' in t and 'NON' not in t and 'NO LEAD' not in t and 'NOT LEAD' not in t:
        return 'lead'
    if 'GALVANIZ' in t:
        return 'galvanized'
    if 'UNKNOWN' in t or 'UNKOWN' in t:
        return 'unknown_plain'
    return 'non_lead_definite'


def entropy(counts):
    p = np.asarray(counts, dtype=float)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum())


def main():
    df = pd.read_parquet(SRC)
    df['loc'] = norm(df['Service Line Locality'])
    df['pubmeth'] = norm(df['Public SL Material Verification Method'])
    df['pubmat'] = df['Current Public Side SL Material'].str.strip()
    df['key'] = df['loc'] + '|' + norm(df['Street Address']) + '|' + df['Zip Code'].str.strip().str[:5]
    df = df[df['pubmat'] != '']                        # the public-side row of each pair
    df = df.drop_duplicates('key')                     # address level

    df['outcome'] = df['pubmat'].map(classify_material)
    fell = df[df['outcome'] == 'non_lead_definite']['pubmat'].value_counts()
    print(f'material strings folded into non_lead_definite: {len(fell)} distinct, '
          f'top: {fell.head(6).to_dict()}')

    broad = df[df['pubmeth'].str.contains(MODEL_RE, na=False)]
    strict = df[df['pubmeth'] == MODEL_STRICT]
    print(f'\nmodel-classified addresses  strict "{MODEL_STRICT}": {len(strict):,}')
    print(f'                            broad  (any predictive/statistical): {len(broad):,}'
          f'   (+{len(broad)-len(strict):,})')
    extra = broad[~broad.index.isin(strict.index)]
    print(f'  the {len(extra):,} extra rows come from: '
          f'{extra["pubmeth"].value_counts().head(5).to_dict()}')
    print(f'  and sit in localities: {extra["loc"].value_counts().head(8).to_dict()}')
    print(f'  ANY of them in NYC? {sorted(set(extra["loc"]) & {"QN","BK","SI","BX","MN"}) or "no"}')

    phys = df[df['pubmeth'].str.contains(PHYS_RE, na=False)]
    prate = phys.groupby('loc')['outcome'].agg(
        phys_n='size', phys_lead=lambda s: int((s == 'lead').sum()))
    prate['phys_lead_pct'] = 100 * prate['phys_lead'] / prate['phys_n']

    rows = []
    for loc, g in broad.groupby('loc'):
        if len(g) < MIN_N:
            continue
        vc = g['outcome'].value_counts()
        raw_vc = g['pubmat'].value_counts()
        p = prate.loc[loc] if loc in prate.index else None
        rows.append(dict(
            locality=loc, n=len(g),
            distinct_raw_values=int(g['pubmat'].nunique()),
            distinct_outcome_classes=int(g['outcome'].nunique()),
            entropy_bits=entropy(raw_vc.values),
            lead_pct=100 * vc.get('lead', 0) / len(g),
            could_be_lead_pct=100 * vc.get('unknown_could_be_lead', 0) / len(g),
            lead_or_hedge_pct=100 * (vc.get('lead', 0) + vc.get('unknown_could_be_lead', 0)) / len(g),
            any_unknown_pct=100 * (len(g) - vc.get('lead', 0) - vc.get('non_lead_definite', 0)
                                   - vc.get('galvanized', 0)) / len(g),
            modal_value=raw_vc.index[0], modal_share=100 * raw_vc.iloc[0] / len(g),
            phys_n=int(p['phys_n']) if p is not None else 0,
            phys_lead_pct=float(p['phys_lead_pct']) if p is not None else float('nan'),
        ))
    t = pd.DataFrame(rows).sort_values('n', ascending=False)
    t['zero_variance'] = t['distinct_raw_values'] == 1
    t['contradicted'] = t['zero_variance'] & (t['lead_or_hedge_pct'] == 0) & \
                        (t['phys_n'] >= 200) & (t['phys_lead_pct'] >= 1.0)

    pd.set_option('display.width', 230); pd.set_option('display.max_rows', 200)
    show = ['locality', 'n', 'distinct_raw_values', 'entropy_bits', 'lead_or_hedge_pct',
            'modal_value', 'modal_share', 'phys_n', 'phys_lead_pct', 'contradicted']
    print(f'\n=== every NY system with >= {MIN_N} model-classified addresses '
          f'({len(t)} systems, {t["n"].sum():,} addresses) ===')
    print(t[show].to_string(index=False, float_format=lambda v: f'{v:,.2f}'))

    zv = t[t['zero_variance']]
    con = t[t['contradicted']]
    print(f'\n--- SCREEN RESULT ---')
    print(f'systems with >= {MIN_N} model-classified addresses : {len(t)}')
    print(f'  reporting exactly ONE distinct material value    : {len(zv)} '
          f'({len(zv)/len(t):.0%}), covering {zv["n"].sum():,} addresses '
          f'({zv["n"].sum()/t["n"].sum():.1%} of all model-classified lines screened)')
    print(f'  ...AND contradicted by their own physical verification '
          f'(>=200 verified, >=1% lead) : {len(con)}, covering {con["n"].sum():,} addresses')
    print(f'\ncontradicted systems:')
    print(con[['locality', 'n', 'modal_value', 'phys_n', 'phys_lead_pct']]
          .to_string(index=False, float_format=lambda v: f'{v:,.2f}'))
    print(f'\nthe mirror cases -- highest model lead-or-hedge rate:')
    print(t.nlargest(8, 'lead_or_hedge_pct')[
        ['locality', 'n', 'lead_pct', 'could_be_lead_pct', 'phys_n', 'phys_lead_pct']]
        .to_string(index=False, float_format=lambda v: f'{v:,.2f}'))

    with open(OUT, 'w') as f:
        json.dump(dict(systems=t.to_dict('records'),
                       n_systems=len(t), n_zero_variance=len(zv),
                       n_contradicted=len(con),
                       addresses_screened=int(t['n'].sum()),
                       addresses_zero_variance=int(zv['n'].sum()),
                       addresses_contradicted=int(con['n'].sum()),
                       strict_n=int(len(strict)), broad_n=int(len(broad))), f,
                  indent=2, default=float)
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
