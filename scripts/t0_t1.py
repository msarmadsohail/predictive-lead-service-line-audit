"""Compare the two snapshots on an address basis."""
import re
import pandas as pd
import paths

W = paths.WORK
MODEL_RE = re.compile(r'PREDICTIVE MODEL|STATISTICAL ANALYSIS')
PHYS_RE = re.compile(r'EXCAVAT|FIELD INSPECT|FIELD INVESTIG|FIELD INSP')
BORO = {'QN', 'BK', 'SI', 'BX', 'MN'}
COLS = ['Service Line Locality', 'Street Address', 'Zip Code',
        'Current Public Side SL Material', 'Public SL Material Verification Method',
        'Customer SL Material', 'Customer SL Material Verification Method']


def norm(s):
    return s.str.strip().str.upper().str.replace(r'\s+', ' ', regex=True)


def prep(path):
    d = pd.read_parquet(path, columns=COLS)
    d['loc'] = norm(d['Service Line Locality'])
    d['key'] = d['loc'] + '|' + norm(d['Street Address']) + '|' + d['Zip Code'].str.strip().str[:5]
    d['pubmeth'] = norm(d['Public SL Material Verification Method'])
    d['cusmeth'] = norm(d['Customer SL Material Verification Method'])
    d['pubmat'] = d['Current Public Side SL Material'].str.strip()
    d['cusmat'] = d['Customer SL Material'].str.strip()
    d['is_model'] = (d['pubmeth'].str.contains(MODEL_RE, na=False) |
                     d['cusmeth'].str.contains(MODEL_RE, na=False))
    d['is_phys'] = (d['pubmeth'].str.contains(PHYS_RE, na=False) |
                    d['cusmeth'].str.contains(PHYS_RE, na=False))
    # model material, either side
    d['model_mat'] = d['pubmat'].where(d['pubmeth'].str.contains(MODEL_RE, na=False), d['cusmat'])
    return d.sort_values('pubmat', ascending=False).drop_duplicates('key')


def flags(mat):
    lead = mat.str.contains('(?i)lead', na=False) & ~mat.str.contains('(?i)non|unlikely', na=False)
    return lead


t0, t1 = prep(W / 't0_state.parquet'), prep(W / 't1_nyc_state.parquet')
print(f'addresses   T0 {len(t0):,}   T1 {len(t1):,}   ({len(t1)-len(t0):+,})\n')

print('=' * 78)
print('STATEWIDE, address basis, model = either side')
print('=' * 78)
for name, d in (('T0', t0), ('T1', t1)):
    m = d[d['is_model']]
    lead = flags(m['model_mat'])
    hedge = m['model_mat'].str.contains('(?i)could be lead', na=False)
    print(f'{name}: addresses {len(d):>9,}   model-classified {len(m):>7,} '
          f'({len(m)/len(d):.2%})   physically verified {int(d["is_phys"].sum()):>7,}')
    print(f'      model says lead {int(lead.sum()):>6,} ({lead.mean():6.2%})   '
          f'hedges could-be-lead {int(hedge.sum()):>6,} ({hedge.mean():6.2%})   '
          f'either {int((lead|hedge).sum()):>6,} ({(lead|hedge).mean():6.2%})')
m0, m1 = t0[t0['is_model']], t1[t1['is_model']]
print(f'\nmodel-classified addresses: {len(m0):,} -> {len(m1):,} '
      f'({len(m1)-len(m0):+,}, {(len(m1)/len(m0)-1)*100:+.1f}% in ~14 months)')
l0, l1 = flags(m0['model_mat']), flags(m1['model_mat'])
print(f'model classifications naming lead: {int(l0.sum()):,} -> {int(l1.sum()):,} '
      f'({int(l1.sum())-int(l0.sum()):+,})')
print(f'  => the model bucket grew by {len(m1)-len(m0):,} addresses and produced '
      f'{int(l1.sum())-int(l0.sum()):+,} additional lead findings.')

print('\n' + '=' * 78)
print('NYC: what changed, and what did not')
print('=' * 78)
for name, d in (('T0', t0), ('T1', t1)):
    n = d[d['loc'].isin(BORO)]
    nm = n[n['is_model']]
    print(f'{name}: NYC addresses {len(n):>8,}   public side populated '
          f'{int((n["pubmat"] != "").sum()):>8,}   model-classified {len(nm):>7,}   '
          f'distinct materials from the model: {nm["model_mat"].nunique()} '
          f'{sorted(nm["model_mat"].unique())}')
n0 = t0[t0['loc'].isin(BORO)]; n1 = t1[t1['loc'].isin(BORO)]
same = set(n0.loc[n0['is_model'], 'key']) & set(n1.loc[n1['is_model'], 'key'])
print(f'\nNYC addresses model-classified at BOTH snapshots: {len(same):,} '
      f'({len(same)/len(n1[n1["is_model"]]):.1%} of T1\'s)')
print(f'  => the zero-variance output is at least 14 months old; it is not a fresh artefact.')

print('\nsystems adopting a predictive model between snapshots:')
s0, s1 = set(m0['loc']), set(m1['loc'])
print(f'  {len(s0)} -> {len(s1)} systems ({len(s1-s0)} new, {len(s0-s1)} stopped)')
if s1 - s0:
    print(f'  new: {m1[m1["loc"].isin(s1-s0)].groupby("loc").size().nlargest(10).to_dict()}')
