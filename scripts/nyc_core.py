"""Loaders and the NYC method table.

Address-level. Label comes from Current Public Side SL Material."""
import re
import numpy as np
import pandas as pd
import paths

PARQUET = paths.WORK / 't1_nyc_state.parquet'
BOROUGHS = ['QN', 'BK', 'BX', 'MN', 'SI']
BOROUGH_NAME = {'QN': 'Queens', 'BK': 'Brooklyn', 'BX': 'Bronx',
                'MN': 'Manhattan', 'SI': 'Staten Island'}
LEAD_MAT = 'Lead including lead-lined galvanized'
POINT_RE = re.compile(r'POINT \(([-\d.]+) ([-\d.]+)\)')
# NYC bounding box
NYC_BBOX = dict(lat_min=40.45, lat_max=40.95, lon_min=-74.30, lon_max=-73.68)


def _norm(s):
    return s.str.strip().str.upper().str.replace(r'\s+', ' ', regex=True)


def load_state():
    df = pd.read_parquet(PARQUET)
    df['loc'] = _norm(df['Service Line Locality'])
    df['pubmeth'] = _norm(df['Public SL Material Verification Method'])
    df['cusmeth'] = _norm(df['Customer SL Material Verification Method'])
    df['pubmat'] = df['Current Public Side SL Material'].str.strip()
    df['cusmat'] = df['Customer SL Material'].str.strip()
    df['cat'] = df['SL Category'].str.strip()
    df['zip5'] = df['Zip Code'].str.strip().str[:5]
    df['key'] = df['loc'] + '|' + _norm(df['Street Address']) + '|' + df['zip5']
    return df


def load_nyc(apply_bbox=True):
    """NYC, address-level: the public-side row of each pair, one row per address.

    `apply_bbox` drops points outside a generous NYC bounding box -- flagged in session 2
    as a known gross-corruption residue and left unfiltered there. Reported either way.
    """
    df = load_state()
    nyc = df[df['loc'].isin(BOROUGHS)].copy()
    n_rows = len(nyc)
    pub = nyc[nyc['pubmat'] != ''].copy()          # the public-side row of each pair
    n_pub = len(pub)
    pub = pub.drop_duplicates('key', keep='first')  # 607 genuine duplicate address keys
    lat, lon = [], []
    for s in pub['Location'].values:
        m = POINT_RE.match(s.strip())
        if m:
            lon.append(float(m.group(1))); lat.append(float(m.group(2)))
        else:
            lon.append(np.nan); lat.append(np.nan)
    pub['lat'] = lat; pub['lon'] = lon
    pub['has_point'] = pub['lat'].notna()
    b = NYC_BBOX
    pub['in_bbox'] = (pub['lat'].between(b['lat_min'], b['lat_max']) &
                      pub['lon'].between(b['lon_min'], b['lon_max']))
    pub['btype'] = pub['Building Type'].str.strip().replace('', '(blank)')
    pub['is_lead'] = pub['pubmat'] == LEAD_MAT
    meta = dict(nyc_rows=n_rows, nyc_public_rows=n_pub, nyc_addresses=len(pub),
                with_point=int(pub['has_point'].sum()),
                out_of_bbox=int((pub['has_point'] & ~pub['in_bbox']).sum()))
    if apply_bbox:
        pub = pub[pub['has_point'] & pub['in_bbox']].copy()
    meta['returned'] = len(pub)
    return pub, meta


def method_table(pub):
    """Public-side outcome by verification method x borough. The article's Table 1."""
    out = []
    for meth in ['RECORDS', 'FIELD INSPECTION', 'EXCAVATION',
                 'STATISTICAL ANALYSIS/PREDICTIVE MODEL', 'NOT VERIFIED']:
        s = pub[pub['pubmeth'] == meth]
        for b in BOROUGHS:
            sb = s[s['loc'] == b]
            if not len(sb):
                continue
            out.append(dict(method=meth, borough=b, n=len(sb),
                            lead=int(sb['is_lead'].sum()),
                            lead_pct=100 * sb['is_lead'].mean(),
                            unknown_pct=100 * sb['pubmat'].str.contains('Unknown').mean()))
        if len(s):
            out.append(dict(method=meth, borough='ALL', n=len(s),
                            lead=int(s['is_lead'].sum()),
                            lead_pct=100 * s['is_lead'].mean(),
                            unknown_pct=100 * s['pubmat'].str.contains('Unknown').mean()))
    return pd.DataFrame(out)


if __name__ == '__main__':
    pd.set_option('display.width', 200); pd.set_option('display.max_rows', 200)
    pub, meta = load_nyc()
    print('NYC record structure:')
    for k, v in meta.items():
        print(f'  {k:>18}: {v:,}')
    print('\n=== Table 1. Public-side lead rate by verification method, NYC ===')
    t = method_table(pub)
    piv = t.pivot(index='method', columns='borough', values='lead_pct').round(2)
    npv = t.pivot(index='method', columns='borough', values='n')
    print('\nlead % of public-side classifications:'); print(piv[BOROUGHS + ['ALL']])
    print('\nn:'); print(npv[BOROUGHS + ['ALL']])
    print('\n=== Model bucket: what value is actually recorded? ===')
    mm = pub[pub['pubmeth'] == 'STATISTICAL ANALYSIS/PREDICTIVE MODEL']
    print(mm['pubmat'].value_counts())
    print('\n=== Same method, rest of the state ===')
    df = load_state()
    rest = df[(~df['loc'].isin(BOROUGHS)) & (df['pubmat'] != '') &
              (df['pubmeth'] == 'STATISTICAL ANALYSIS/PREDICTIVE MODEL')]
    rest = rest.drop_duplicates('key')
    vc = rest['pubmat'].value_counts()
    print(vc)
    hedge = vc.filter(regex='(?i)could be lead').sum()
    lead = vc.filter(regex='(?i)^lead').sum()
    print(f"\nrest-of-state model addresses: {len(rest):,}")
    print(f"  says LEAD outright          : {lead:,} ({lead/len(rest):.2%})")
    print(f"  hedges 'could be lead'      : {hedge:,} ({hedge/len(rest):.2%})")
    print(f"  either                      : {lead+hedge:,} ({(lead+hedge)/len(rest):.2%})")
    print(f"NYC model addresses           : {len(mm):,}")
    print(f"  says LEAD or hedges         : {int(mm['pubmat'].str.contains('(?i)lead').sum()):,} (0.00%)")
