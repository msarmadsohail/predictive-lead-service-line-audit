"""Verification volume by system."""
import re
import numpy as np
import pandas as pd
from scipy import stats
import paths

W = paths.WORK
MODEL_RE = re.compile(r'PREDICTIVE MODEL|STATISTICAL ANALYSIS')
PHYS_RE = re.compile(r'EXCAVAT|FIELD INSPECT|FIELD INVESTIG|FIELD INSP')
BORO = {'QN', 'BK', 'SI', 'BX', 'MN'}


def norm(s):
    return s.str.strip().str.upper().str.replace(r'\s+', ' ', regex=True)


def prep(path, cols):
    d = pd.read_parquet(path, columns=cols)
    d['loc'] = norm(d['Service Line Locality'])
    d['pubmeth'] = norm(d['Public SL Material Verification Method'])
    d['pubmat'] = d['Current Public Side SL Material'].str.strip()
    d['key'] = d['loc'] + '|' + norm(d['Street Address']) + '|' + d['Zip Code'].str.strip().str[:5]
    return d[d['pubmat'] != ''].drop_duplicates('key')


COLS = ['Service Line Locality', 'Street Address', 'Zip Code',
        'Current Public Side SL Material', 'Public SL Material Verification Method']

print('=' * 78)
print('(a) DOES LEANING ON THE MODEL GO WITH VERIFYING LESS?')
print('=' * 78)
t1 = prep(W / 't1_nyc_state.parquet', COLS)
g = t1.groupby('loc').agg(total=('key', 'size'))
g['model'] = t1[t1['pubmeth'].str.contains(MODEL_RE, na=False)].groupby('loc').size()
g['phys'] = t1[t1['pubmeth'].str.contains(PHYS_RE, na=False)].groupby('loc').size()
g = g.fillna(0)
g = g[g['model'] >= 100].copy()
g['model_share'] = 100 * g['model'] / g['total']
g['phys_share'] = 100 * g['phys'] / g['total']
g['verified_per_modelled'] = g['phys'] / g['model']
r = stats.spearmanr(g['model_share'], g['phys_share'])
print(f'{len(g)} systems with >=100 model-classified addresses')
print(f'Spearman(model share of inventory, physical-verification share) = '
      f'{r.statistic:.3f}  p = {r.pvalue:.2g}')
print(f'median physically-verified lines per model-classified line: '
      f'{g["verified_per_modelled"].median():.2f}')
print(f'  systems with ZERO physical verification at all: '
      f'{int((g["phys"] == 0).sum())} of {len(g)}, covering '
      f'{int(g.loc[g["phys"] == 0, "model"].sum()):,} model-classified addresses')
print('\nlargest model deployments and how checkable each is:')
print(g.nlargest(12, 'model')[['total', 'model', 'phys', 'model_share', 'verified_per_modelled']]
      .to_string(float_format=lambda v: f'{v:,.2f}'))

print('\n' + '=' * 78)
print('(b) T0 (2025-06-22) vs T1 (2026-08-11): is the model bucket growing?')
print('=' * 78)
t0 = prep(W / 't0_state.parquet', COLS)
print(f'addresses  T0 {len(t0):,}  ->  T1 {len(t1):,}   ({len(t1)-len(t0):+,})')
for name, d in (('T0', t0), ('T1', t1)):
    m = d[d['pubmeth'].str.contains(MODEL_RE, na=False)]
    p = d[d['pubmeth'].str.contains(PHYS_RE, na=False)]
    lead = m['pubmat'].str.contains('(?i)lead', na=False) & \
        ~m['pubmat'].str.contains('(?i)non|unlikely', na=False)
    print(f'{name}: model-classified {len(m):>7,} ({len(m)/len(d):.2%} of inventory)   '
          f'physically verified {len(p):>7,}   model rows mentioning lead/could-be-lead '
          f'{int(lead.sum()):>6,} ({lead.mean():.2%})')
m0 = t0[t0['pubmeth'].str.contains(MODEL_RE, na=False)]
m1 = t1[t1['pubmeth'].str.contains(MODEL_RE, na=False)]
print(f'\nmodel-classified growth: {len(m0):,} -> {len(m1):,} '
      f'({(len(m1)/max(len(m0),1)-1)*100:+.1f}% in ~14 months)')

print('\nNYC specifically:')
for name, d in (('T0', t0), ('T1', t1)):
    n = d[d['loc'].isin(BORO)]
    nm = n[n['pubmeth'].str.contains(MODEL_RE, na=False)]
    print(f'  {name}: NYC addresses {len(n):>8,}   model-classified {len(nm):>7,}   '
          f'distinct recorded materials from the model: {nm["pubmat"].nunique()}  '
          f'{sorted(nm["pubmat"].unique())[:4]}')

print('\nsystems newly using a predictive model between T0 and T1:')
s0 = set(m0['loc'].unique()); s1 = set(m1['loc'].unique())
new = sorted(s1 - s0)
print(f'  {len(s0)} systems at T0 -> {len(s1)} at T1;  {len(new)} new, {len(s0-s1)} stopped')
newn = m1[m1['loc'].isin(new)].groupby('loc').size().nlargest(10)
print(f'  largest new adopters: {newn.to_dict()}')
