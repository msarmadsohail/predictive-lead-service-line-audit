"""Figures 2-4."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import paths

W = paths.WORK
F = paths.FIG
N = json.load(open(W / 'paper_numbers.json'))
E = json.load(open(W / 'estimate_results.json'))

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': .25, 'grid.linewidth': .5,
    'axes.axisbelow': True, 'figure.dpi': 200,
})
INK, RED, BLUE, GREY = '#1a1a1a', '#c0392b', '#2c5f8a', '#8a8a8a'

# Figure 1
rv = N['records_vs_model']
def lab(s):
    a, b = s.strip('[)').split(', ')
    return f'{a}-{b}'
eras = [lab(r['era']) for r in rv]
x = np.arange(len(eras)); w = .27
fig, ax = plt.subplots(figsize=(6.6, 2.9))
ax.bar(x - w, [r['records lead %'] for r in rv], w, label='Records (desk method)',
       color=BLUE, edgecolor='none')
ax.bar(x, [r['physical lead %'] for r in rv], w, label='Excavation or field inspection',
       color=GREY, edgecolor='none')
ax.bar(x + w, [r['model lead %'] for r in rv], w, label='Predictive model (desk method)',
       color=RED, edgecolor='none')
for i, r in enumerate(rv):
    ax.plot([i + w - w/2, i + w + w/2], [0, 0], color=RED, lw=1.6, solid_capstyle='butt',
            zorder=4, clip_on=False)
    ax.annotate(f"0 of {r['model n']:,}", (i + w, .7), rotation=90, ha='center',
                va='bottom', fontsize=5.8, color=RED)
ax.set_xticks(x); ax.set_xticklabels(eras, fontsize=7.5)
ax.set_ylabel('public-side classifications recorded as lead (%)')
ax.set_xlabel('year the building was built (NYC MapPLUTO)')
ax.legend(frameon=False, fontsize=7.6, loc='upper right')
ax.set_ylim(0, 37)
fig.tight_layout(); fig.savefig(F / 'fig1_era.pdf'); plt.close(fig)

# Figure 2
rows = E['rows']
ERA_AWARE = {'A3 borough x era', 'A4 borough x type x era', 'A5 ZIP x era',
             'B500e spatial 500 m + era', 'B1000e spatial 1000 m + era',
             'C2 GBM + PLUTO era & built form'}
rows = sorted(rows, key=lambda r: r['implied_extrapolated'])
names = [r['estimator'].split(' ', 1)[1] for r in rows]
vals = [r['implied_extrapolated'] for r in rows]
aware = [r['estimator'] in ERA_AWARE for r in rows]
fig, ax = plt.subplots(figsize=(6.6, 3.2))
ea = [v for v, a in zip(vals, aware) if a]
ax.axvspan(min(ea), max(ea), color=RED, alpha=.09, lw=0)
for i, (v, a, r) in enumerate(zip(vals, aware, rows)):
    ax.plot([0, v], [i, i], color='#dcdcdc', lw=.8, zorder=1)
    ax.scatter(v, i, s=34, color=RED if a else BLUE, zorder=3,
               marker='o' if a else 's')
    ax.annotate(f"{v:,.0f}", (v, i), xytext=(7, 0), textcoords='offset points',
                va='center', fontsize=7.4,
                color=RED if a else BLUE)
    if r['coverage'] < .999:
        ax.annotate(f"{r['coverage']:.0%} cov.", (0, i), xytext=(4, 0),
                    textcoords='offset points', va='center', fontsize=6, color=GREY)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=7.4)
ax.set_xlim(0, 3150)
ax.set_xlabel('expected lead public-side lines among the 43,215 model-cleared addresses')
h = [plt.Line2D([], [], marker='s', ls='', color=BLUE, label='construction era not controlled'),
     plt.Line2D([], [], marker='o', ls='', color=RED, label='construction era controlled')]
ax.legend(handles=h, frameon=False, fontsize=7.6, loc='lower right')
fig.tight_layout(); fig.savefig(F / 'fig2_estimators.pdf'); plt.close(fig)

# Figure 3
rest = N['rest_of_state']
order = ['recorded as known non-lead\n(Known Other, Copper, Plastic, Galvanized)',
         'unknown, no lead judgement',
         'unknown but could be lead',
         'lead']
rest_n, rest_lead, rest_hedge = rest['n'], rest['lead'], rest['hedge']
rest_nonlead = 106796 + 24959 + 38 + 4182 + 1137
rest_unk = rest_n - rest_lead - rest_hedge - rest_nonlead
data = {'Rest of New York State\n(176,888 addresses)':
        [rest_nonlead / rest_n, rest_unk / rest_n, rest_hedge / rest_n, rest_lead / rest_n],
        'New York City\n(43,215 addresses)': [1.0, 0, 0, 0]}
cols = ['#cfd8e0', '#9aa8b4', '#e8a33d', RED]
fig, ax = plt.subplots(figsize=(6.6, 2.15))
for j, (k, v) in enumerate(data.items()):
    left = 0
    for val, c, name in zip(v, cols, order):
        if val <= 0:
            continue
        ax.barh(j, val * 100, left=left * 100, height=.52, color=c,
                edgecolor='white', lw=.7, label=name if j == 0 else None)
        mid = left * 100 + val * 50
        if val >= .05:
            ax.annotate(f'{val*100:.1f}%', (mid, j), ha='center', va='center',
                        fontsize=7.4, color='white' if c == RED else INK)
        else:
            ax.annotate(f'{val*100:.1f}%', (mid, j + .40), ha='center', va='bottom',
                        fontsize=6.6, color=c)
        left += val
ax.set_yticks([0, 1]); ax.set_yticklabels(list(data), fontsize=8)
ax.set_ylim(-.62, 1.62)
ax.set_xlim(0, 100)
ax.set_xlabel('share of addresses classified by a predictive model (%)', labelpad=4)
ax.grid(axis='y', visible=False)
fig.subplots_adjust(bottom=.42, left=.24, right=.98, top=.96)
fig.legend(frameon=False, fontsize=6.9, ncol=2, loc='lower center',
           bbox_to_anchor=(.55, -.02), handlelength=1.1, columnspacing=1.4)
fig.savefig(F / 'fig3_vocabulary.pdf'); plt.close(fig)

print('wrote fig1_era.pdf, fig2_estimators.pdf, fig3_vocabulary.pdf')
print(f"  sanity: rest-of-state shares sum to "
      f"{(rest_nonlead+rest_unk+rest_hedge+rest_lead)/rest_n:.4f}")
