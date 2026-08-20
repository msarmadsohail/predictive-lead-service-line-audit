"""Emit every figure cited in the manuscript."""
import json
import numpy as np
import pandas as pd
from scipy import stats

import nyc_core as C
import paths

JOINED = paths.WORK / 'nyc_joined.parquet'
MODEL_METH = 'STATISTICAL ANALYSIS/PREDICTIVE MODEL'
PHYS = ['EXCAVATION', 'FIELD INSPECTION']
ERA_BINS = [1800, 1900, 1920, 1940, 1950, 1960, 1980, 2000, 2027]
OUT = {}

d = pd.read_parquet(JOINED)
d['era'] = pd.cut(d['pluto_yearbuilt'], ERA_BINS, right=False)
phys = d[d['pubmeth'].isin(PHYS) & d['pubmat'].isin([C.LEAD_MAT, 'Known Other'])]
cleared = d[d['pubmeth'] == MODEL_METH]

print('=' * 78)
print('1. THE ZERO-VARIANCE FACT')
print('=' * 78)
mm = d[d['pubmeth'] == MODEL_METH]
OUT['nyc_model_addresses'] = int(len(mm))
OUT['nyc_model_values'] = mm['pubmat'].value_counts().to_dict()
print(f'NYC addresses classified by predictive model : {len(mm):,}')
print(f'distinct recorded materials                  : {mm["pubmat"].unique().tolist()}')
hi = stats.beta.ppf(0.975, 0 + 1, len(mm))          # 95% one-sided upper bound, 0 events
OUT['nyc_model_lead_upper95'] = float(hi)
print(f'95% upper bound on the NYC model lead rate   : {hi:.6%}  (0 of {len(mm):,})')

state = C.load_state()
rest = state[(~state['loc'].isin(C.BOROUGHS)) & (state['pubmat'] != '') &
             (state['pubmeth'] == MODEL_METH)].drop_duplicates('key')
vc = rest['pubmat'].value_counts()
lead_out = int(vc.filter(regex='(?i)^lead').sum())
hedge = int(vc.filter(regex='(?i)could be lead').sum())
OUT['rest_of_state'] = dict(n=int(len(rest)), lead=lead_out, hedge=hedge,
                            lead_pct=100 * lead_out / len(rest),
                            hedge_pct=100 * hedge / len(rest),
                            either_pct=100 * (lead_out + hedge) / len(rest))
print(f'\nsame method, rest of New York State          : {len(rest):,} addresses')
print(f'  recorded LEAD                              : {lead_out:,} ({lead_out/len(rest):.2%})')
print(f'  recorded "unknown but could be lead"       : {hedge:,} ({hedge/len(rest):.2%})')
print(f'  either                                     : {lead_out+hedge:,} '
      f'({(lead_out+hedge)/len(rest):.2%})')
p_rest = (lead_out + hedge) / len(rest)
print(f'  P(0 of {len(mm):,} | rate = rest-of-state)   : {np.exp(len(mm)*np.log1p(-p_rest)):.3g}')
print(f'  ratio, rest-of-state rate / NYC 95% bound  : {p_rest/hi:,.0f}x')
OUT['ratio_rest_to_nyc_bound'] = float(p_rest / hi)

print('\n' + '=' * 78)
print('2. TABLE 1 -- five methods, one city')
print('=' * 78)
t = []
for meth in ['RECORDS', 'EXCAVATION', 'FIELD INSPECTION', MODEL_METH, 'NOT VERIFIED']:
    s = d[d['pubmeth'] == meth]
    t.append(dict(method=meth, n=len(s), share=100 * len(s) / len(d),
                  lead=int(s['is_lead'].sum()), lead_pct=100 * s['is_lead'].mean(),
                  unknown_pct=100 * s['pubmat'].str.contains('Unknown').mean(),
                  median_yearbuilt=float(s['pluto_yearbuilt'].median())))
t1 = pd.DataFrame(t)
print(t1.to_string(index=False, float_format=lambda v: f'{v:,.2f}'))
OUT['table1'] = t1.to_dict('records')

print('\n' + '=' * 78)
print('3. LEAD RATE BY CONSTRUCTION ERA (physically verified) AND THE ERA MIX')
print('=' * 78)
e = phys.groupby('era', observed=True)['is_lead'].agg(n='size', lead='sum')
e['lead_pct'] = 100 * e['lead'] / e['n']
e['model_cleared_n'] = cleared.groupby('era', observed=True).size()
e['phys_share'] = 100 * e['n'] / e['n'].sum()
e['cleared_share'] = 100 * e['model_cleared_n'] / e['model_cleared_n'].sum()
e['implied'] = e['lead_pct'] / 100 * e['model_cleared_n']
print(e.to_string(float_format=lambda v: f'{v:,.2f}'))
OUT['era_table'] = e.reset_index().astype({'era': str}).to_dict('records')
print(f'\nmedian year built  physically verified {phys["pluto_yearbuilt"].median():.0f}  '
      f'model-cleared {cleared["pluto_yearbuilt"].median():.0f}')
print(f'era-stratified implied total (borough-blind): {e["implied"].sum():,.0f}')

print('\n' + '=' * 78)
print('4. THE SUBGROUP THAT NEEDS NO MODEL: pre-1940 buildings')
print('=' * 78)
pre = cleared[cleared['pluto_yearbuilt'] < 1940]
pre_ph = phys[phys['pluto_yearbuilt'] < 1940]
r = pre_ph['is_lead'].mean()
ci = stats.binomtest(int(pre_ph['is_lead'].sum()), len(pre_ph)).proportion_ci(0.95)
print(f'model-cleared addresses in pre-1940 buildings      : {len(pre):,} '
      f'({len(pre)/len(cleared):.1%} of the cleared population)')
print(f'physically-verified lead rate, pre-1940, same city : {r:.2%} '
      f'(95% CI {ci.low:.2%}-{ci.high:.2%}, n={len(pre_ph):,})')
print(f'implied lead lines in this subgroup alone          : {r*len(pre):,.0f} '
      f'({ci.low*len(pre):,.0f}-{ci.high*len(pre):,.0f})')
OUT['pre1940'] = dict(cleared_n=int(len(pre)), phys_n=int(len(pre_ph)), rate=float(r),
                      ci=[float(ci.low), float(ci.high)], implied=float(r * len(pre)))
by_b = pre.groupby('loc').size()
print('\n  by borough:', {C.BOROUGH_NAME[k]: int(v) for k, v in by_b.items()})

print('\n' + '=' * 78)
print('5. INTERNAL CONTROL -- the other desk method finds lead constantly')
print('=' * 78)
rec = d[d['pubmeth'] == 'RECORDS']
cmp_ = pd.DataFrame({
    'records n': rec.groupby('era', observed=True).size(),
    'records lead %': 100 * rec.groupby('era', observed=True)['is_lead'].mean(),
    'model n': cleared.groupby('era', observed=True).size(),
    'model lead %': 100 * cleared.groupby('era', observed=True)['is_lead'].mean(),
    'physical lead %': 100 * phys.groupby('era', observed=True)['is_lead'].mean()})
print(cmp_.to_string(float_format=lambda v: f'{v:,.2f}'))
OUT['records_vs_model'] = cmp_.reset_index().astype({'era': str}).to_dict('records')

print('\n' + '=' * 78)
print('6. EXCAVATION vs FIELD INSPECTION -- are the two "physical" methods the same?')
print('=' * 78)
for m in PHYS:
    s = d[(d['pubmeth'] == m) & d['pubmat'].isin([C.LEAD_MAT, 'Known Other'])]
    su = d[d['pubmeth'] == m]
    print(f'  {m:<18} n={len(su):>7,}  lead {s["is_lead"].mean():6.2%}  '
          f'recorded Unknown despite the method: {su["pubmat"].str.contains("Unknown").sum():>5,} '
          f'({su["pubmat"].str.contains("Unknown").mean():.2%})   median built '
          f'{su["pluto_yearbuilt"].median():.0f}')
ex = d[(d['pubmeth'] == 'EXCAVATION') & d['pubmat'].isin([C.LEAD_MAT, 'Known Other'])]
fi = d[(d['pubmeth'] == 'FIELD INSPECTION') & d['pubmat'].isin([C.LEAD_MAT, 'Known Other'])]
z = stats.chi2_contingency([[ex['is_lead'].sum(), len(ex) - ex['is_lead'].sum()],
                            [fi['is_lead'].sum(), len(fi) - fi['is_lead'].sum()]])
print(f'  chi2 p = {z.pvalue:.3g} -- the two are NOT interchangeable; reported separately')
OUT['exc_vs_fi_p'] = float(z.pvalue)

print('\n' + '=' * 78)
print('7. QUANTITIES CITED IN THE TEXT BUT NOT PREVIOUSLY EMITTED')
print('=' * 78)
MODEL_RE = r'PREDICTIVE MODEL|STATISTICAL ANALYSIS'
PHYS_RE = r'EXCAVAT|FIELD INSPECT|FIELD INVESTIG|FIELD INSP'
addr = state[state['pubmat'] != ''].drop_duplicates('key')
T = {}

# trap 3: category disagrees with the public-side material
from statewide_screen import classify_material
addr_out = addr['pubmat'].map(classify_material)
T['cat_lead_material_not_lead'] = int(((addr['cat'] == 'Lead') &
                                       (addr_out == 'non_lead_definite')).sum())

# trap 4: what broad method matching costs the rest-of-state comparison
broad = addr[(~addr['loc'].isin(C.BOROUGHS)) &
             addr['pubmeth'].str.contains(MODEL_RE, na=False)]
bl = int(broad['pubmat'].value_counts().filter(regex='(?i)^lead').sum())
bh = int(broad['pubmat'].str.contains('could be lead', case=False, na=False).sum())
T['rest_broad'] = dict(n=len(broad), either_pct=100 * (bl + bh) / len(broad))
T['rest_strict_dedup_first'] = int(len(addr[(~addr['loc'].isin(C.BOROUGHS)) &
                                            (addr['pubmeth'] == MODEL_METH)]))

# the Poughkeepsie concentration inside the rest-of-state lead component
pk = rest[rest['loc'] == 'CITY OF POUGHKEEPSIE']
pk_lead = int(pk['pubmat'].value_counts().filter(regex='(?i)^lead').sum())
nopk = rest[rest['loc'] != 'CITY OF POUGHKEEPSIE']
nl = int(nopk['pubmat'].value_counts().filter(regex='(?i)^lead').sum())
nh = int(nopk['pubmat'].str.contains('could be lead', case=False, na=False).sum())
T['poughkeepsie'] = dict(n=len(pk), lead=pk_lead, lead_pct=100 * pk_lead / len(pk),
                         share_of_rest_lead=100 * pk_lead / lead_out,
                         phys_n=int(addr[(addr['loc'] == 'CITY OF POUGHKEEPSIE') &
                                         addr['pubmeth'].str.contains(PHYS_RE, na=False)]
                                    .shape[0]))
T['rest_excl_poughkeepsie'] = dict(n=len(nopk), lead_pct=100 * nl / len(nopk),
                                   either_pct=100 * (nl + nh) / len(nopk))

# NYC vocabulary, and the Unknown counts the within-city argument rests on
nyc_all = addr[addr['loc'].isin(C.BOROUGHS)]
T['nyc_vocabulary_all_methods'] = sorted(nyc_all['pubmat'].unique().tolist())
unk = d[d['pubmat'].str.contains('Unknown', case=False, na=False)]
T['nyc_unknown'] = dict(total=int(len(unk)),
                        by_method=unk['pubmeth'].value_counts().to_dict())
T['nyc_lead_addresses'] = int(d['is_lead'].sum())

# per-locality contradiction strength, and the water-system unit
scr = json.load(open(paths.WORK / 'statewide_screen.json'))
T['contradicted_pzero'] = {
    s['locality']: dict(n=s['n'], phys_n=s['phys_n'], phys_lead_pct=s['phys_lead_pct'],
                        expected=s['phys_lead_pct'] / 100 * s['n'],
                        log10_p=float(s['n'] * np.log1p(-s['phys_lead_pct'] / 100)
                                      / np.log(10)))
    for s in scr['systems'] if s['contradicted']}

# addresses carrying both a model row and a physical row
st = state.copy()
st['m'] = st['pubmeth'].str.contains(MODEL_RE, na=False)
st['p'] = st['pubmeth'].str.contains(PHYS_RE, na=False)
gk = st.groupby('key').agg(m=('m', 'max'), p=('p', 'max'))
both = set(gk[gk['m'] & gk['p']].index)
T['both_predicted_and_verified'] = dict(
    statewide=len(both),
    nyc=len(both & set(st.loc[st['loc'].isin(C.BOROUGHS), 'key'])))

# exclusions, and the duplicate-key rule
pub_nb, meta_nb = C.load_nyc(apply_bbox=False)
mnb = pub_nb[pub_nb['pubmeth'] == MODEL_METH]
T['nyc_exclusions'] = dict(no_coordinate=int(meta_nb['nyc_addresses'] - meta_nb['with_point']),
                           out_of_bbox=int(meta_nb['out_of_bbox']),
                           model_no_coordinate=int((~mnb['has_point']).sum()),
                           model_out_of_bbox=int((mnb['has_point'] & ~mnb['in_bbox']).sum()),
                           model_rows_raw=int(len(mnb)),
                           dropped_duplicate_keys=int(meta_nb['nyc_public_rows']
                                                      - meta_nb['nyc_addresses']))

# year-built heaping
yb = d.loc[d['pluto_yearbuilt'] > 0, 'pluto_yearbuilt']
T['yearbuilt_heaping'] = dict(multiple_of_10=100 * float((yb % 10 == 0).mean()),
                              multiple_of_5=100 * float((yb % 5 == 0).mean()))
T['cleared_post_1980_pct'] = 100 * float((cleared['pluto_yearbuilt'] >= 1980).mean())

for k, v in T.items():
    print(f'  {k}: {v}')
OUT['text_quantities'] = T

with open(paths.WORK / 'paper_numbers.json', 'w') as f:
    json.dump(OUT, f, indent=2, default=str)
print('\nwrote work/paper_numbers.json')
