"""Figure 1."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import paths

S = json.load(open(paths.WORK / 'statewide_screen.json'))
F = paths.FIG
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': .25, 'grid.linewidth': .5,
                     'axes.axisbelow': True, 'figure.dpi': 200})
INK, RED, BLUE, GREY, AMB = '#1a1a1a', '#c0392b', '#2c5f8a', '#9aa8b4', '#e8a33d'
sy = S['systems']
n = np.array([s['n'] for s in sy], float)
y = np.array([s['lead_or_hedge_pct'] for s in sy], float)
zv = np.array([s['zero_variance'] for s in sy], bool)
con = np.array([s['contradicted'] for s in sy], bool)

fig, ax = plt.subplots(figsize=(6.6, 3.5))
ax.scatter(n[~zv], y[~zv], s=17, c=BLUE, alpha=.65, lw=0,
           label=f'output varies ({int((~zv).sum())} systems)')
ax.scatter(n[zv & ~con], y[zv & ~con], s=22, facecolors='none', edgecolors=GREY, lw=.9,
           label=f'single recorded value, not contradicted ({int((zv & ~con).sum())})')
ax.scatter(n[con], y[con], s=52, c=RED, marker='D', lw=0,
           label=f'single value, contradicted by the system\'s own crews ({int(con.sum())})')

LBL = {'QN': 'Queens', 'BK': 'Brooklyn', 'SI': 'Staten I.', 'BX': 'Bronx', 'MN': 'Manhattan',
       'EAST ROCHESTER': 'East Rochester', 'CITY OF POUGHKEEPSIE': 'Poughkeepsie',
       'DEWITT': 'DeWitt', 'GREECE': 'Greece', 'AMSTERDAM': 'Amsterdam', 'TROY': 'Troy'}
OFF = {'MN': (-32, 7), 'BX': (-10, 18), 'QN': (-18, 7), 'BK': (0, 18), 'SI': (-6, 7),
       'GREECE': (10, 18), 'EAST ROCHESTER': (-48, 8), 'TROY': (-9, 17),
       'CITY OF POUGHKEEPSIE': (-16, 8), 'DEWITT': (-36, -12), 'AMSTERDAM': (9, 2)}
for s in sy:
    k = s['locality']
    if k in LBL:
        c = RED if s['contradicted'] else (INK if s['lead_or_hedge_pct'] > 40 else GREY)
        ax.annotate(LBL[k], (s['n'], s['lead_or_hedge_pct']), xytext=OFF[k],
                    textcoords='offset points', fontsize=6.6, color=c)
ax.set_xscale('log')
ax.set_xlabel('addresses the system classified with a predictive model (log scale)')
ax.set_ylabel('of those, % recorded as lead\nor as "unknown but could be lead"')
ax.set_ylim(-6, 108)
ax.legend(frameon=False, fontsize=7, loc='upper center', bbox_to_anchor=(.5, 1.19), ncol=1,
          handletextpad=.4, borderpad=0)
fig.tight_layout()
fig.savefig(F / 'fig1_screen.pdf', bbox_inches='tight')
print('wrote fig1_screen.pdf  |  systems', len(sy),
      ' zero-variance', int(zv.sum()), ' contradicted', int(con.sum()))
