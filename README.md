# Auditing recorded predictive lead service-line classifications against physical verification

[![arXiv](https://img.shields.io/badge/arXiv-2608.19922-b31b1b.svg)](https://arxiv.org/abs/2608.19922)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22039275.svg)](https://doi.org/10.5281/zenodo.22039275)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Analysis code and data provenance for a statewide audit of New York State's public Lead Service Line
Inventory.

**Paper:** [arXiv:2608.19922](https://arxiv.org/abs/2608.19922), CC BY 4.0.
**Cite the code as** `10.5281/zenodo.22039275` — the concept DOI, which always resolves to the
newest release. A version-specific DOI is also minted per release and freezes on that tag.

Under the US Lead and Copper Rule Revisions a water utility may determine a service line's material
with a statistical or predictive model instead of digging it up. New York publishes, per address,
which method was used, which makes those model outputs checkable against physical verification
carried out by the same utilities in the same places. This repository holds the screen, the
estimators, an open baseline, and the manuscript.

## Reproducing

```
python3 -m venv .venv && .venv/bin/pip install pandas numpy scikit-learn scipy pyarrow matplotlib
cd ny-lsl-inventory && sha256sum -c SHA256SUMS      # after fetching the captures, see below
cd ../scripts
../.venv/bin/python extract.py          # gz  -> work/t1_nyc_state.parquet
../.venv/bin/python extract_t0.py       # gz  -> work/t0_state.parquet
../.venv/bin/python fetch_mappluto.py   # NYC Open Data -> work/mappluto.json.gz
../.venv/bin/python join_pluto.py       # spatial join  -> work/nyc_joined.parquet
../.venv/bin/python statewide_screen.py # the screen
../.venv/bin/python t0_t1.py            # snapshot comparison
../.venv/bin/python estimate.py         # eleven estimators
../.venv/bin/python baseline2.py        # baseline classifier, CV sensitivity, calibration
../.venv/bin/python paper_numbers.py    # every quantity the manuscript cites
../.venv/bin/python make_figures.py && ../.venv/bin/python make_fig_screen.py
cd ../paper && make            # builds main.pdf
```

Paths resolve from the repository root via `scripts/paths.py`. Set `LSL_ROOT` to run the scripts from
anywhere else, which is what the ancillary bundle attached to the preprint needs.

## The source captures are not committed

**The `.gz` files in `ny-lsl-inventory/` are deliberately not in this repository.** They are 180 MB
together and one is 99 MB, which would permanently bloat the history. **This file plus `SHA256SUMS` is
what makes them reproducible: the bytes are not the artefact, the provenance is.** T1 can be
re-fetched from the endpoint below and checked against the recorded digest. T0 comes from the Internet
Archive and may not be re-fetchable.

## What is here

| File | Size | Rows (API count) | Re-obtainable? |
|---|---|---|---|
| `ny-lsl-inventory/t0_20250622.csv.gz` | 81 MB | 3,747,025 | **NO - irreplaceable in practice** |
| `ny-lsl-inventory/t1_20260811.csv.gz` | 99 MB | 4,618,115 | Yes, but it is a moving target |
| `ny-lsl-inventory/SHA256SUMS` | — | — | checksums for both |
| `scripts/` | small | — | the analysis chain and the Socrata view metadata behind Trap 5 |

### T1 — live snapshot, taken 2026-08-11

New York State Lead Service Line Inventory, Socrata dataset **`j63k-4n92`**, host
**`health.data.ny.gov`** (`data.ny.gov` 302-redirects and returns 0 bytes — request the health host
directly). No authentication.

    curl -s "https://health.data.ny.gov/resource/j63k-4n92.csv?\$limit=5000000" -o t1.csv

It is re-downloadable, but **the live dataset changes**, so this exact capture is what pins every
number in the article. Keep it.

### T0 — archived snapshot, 2025-06-22, via the Wayback Machine

**This is the only known bulk capture of the earlier state of the inventory.** It is what the
archived-snapshot design was tested against, and Wayback may rate-limit or fail on a re-fetch. If it is
lost, that measurement cannot be re-run. **Treat it as primary evidence, not as cache.**

## Traps found the hard way — read before writing any parsing code

1. **Use a real CSV parser. Never split on newlines.** `wc -l` returns 3,775,497 and 4,646,587 against
   API counts of 3,747,025 and 4,618,115 — **exactly 28,472 extra physical lines in each file**,
   from embedded newlines inside quoted fields (the `note` column is the likely source). Naive line
   splitting silently corrupts the join.
2. **No row identifier exists.** The only usable key is `locality + street_address + zip_code`, which is
   **98.97 %** unique at T0. Duplicate keys are real (multiple service lines at one address) and must
   not be dropped silently — dropping them is what hid the 65 re-verification cases on the first pass.
3. **`locality` is case-split.** 4,548 raw strings collapse to **3,974**; **528 groups differ by case
   alone** (`Buffalo` 87,746 vs `BUFFALO` 8,336). **Case-fold every locality cut or it undercounts.**
4. **Schema drift between snapshots:** T0 has 19 columns, T1 has 20 — `State` was added.
5. **Truncated aggregations are not zeros.** A Socrata `$group` under a `$limit` returns only the top
   groups; absent groups are *unmeasured*, not zero. This produced a table of false zeros that had
   to be corrected.
6. **Dirty values are load-bearing, not noise.** `sl_category` contains `Err:508` (3,163) and `#REF!`,
   six spellings of "non-lead", and `Records`/`records`/`RECORDS` at 2,244,787/13,127/3,365. Normalise
   for analysis, **but keep the raw counts — they are a published result.**
7. **NYC's five boroughs are not spelled out in `locality`.** They are two-letter agency codes: `QN`
   `BK` `SI` `BX` `MN`. A literal cut on `"Queens"` returns 11 rows statewide, not 8,515 — enough to
   wrongly conclude NYC is barely represented. Found session 2, 2026-08-18, by cross-checking against
   ZIP code. `Public SL Material Verification Method` also has the same free-text pollution as
   `locality` — e.g. `excavation`/`Excavation at 1 location`/`Mechanical excavation at one location` vs
   canonical `Excavation` — measured at +443 (model) / +631 (physical) rows statewide under a loose
   case/whitespace fold. The screen matches the method broadly; the rest-of-state comparison uses
   the canonical spelling, and the manuscript reports what that choice costs.

8. **NYC files two rows per address — and it did NOT at T0.** Found session 3, 2026-08-18. At T1 all
   five boroughs sit at a rows-per-address ratio of **2.001**; no other locality above 5,000 rows
   exceeds 1.14. One row of each pair carries the public columns + `SL Category`, the other only the
   customer columns; `Location` is identical on 100.00% of pairs, and on the public-side row public
   method == customer method and public material == customer material == `SL Category` on 100.0%.
   **At T0 (2025-06-22) the ratio is 1.001 and the public side is blank on all 817,982 NYC rows** —
   including all 43,440 model classifications, which exist only on the CUSTOMER side, all
   `Known Other`. **The public-side row was added between the snapshots, copying customer-side values
   that already existed.** Counting T1 NYC rows as service lines double-counts; counting
   "public + customer" triple-counts. Statewide, 4,618,115 T1 rows are **3,744,223 distinct keys**.
   **And any T0/T1 diff that filters on a populated public side silently reports NYC as absent from
   T0. It is not — 817,982 rows of it. Diff on an address key, taking the method from either side.**
9. **Read the material, not the category.** `SL Category` collapses to `Lead / Non-Lead / GSLRR /
   Unknown` and, **outside NYC, reports the worse of the two sides** — 9,625 addresses carry a non-lead
   `Current Public Side SL Material` and `SL Category = Lead`, driven by the customer side. Inside NYC
   the two coincide exactly (trap #8), which is why NYC numbers read off `SL Category` survived.
   `Current Public Side SL Material` is the real field and it is much richer: `Known Other`, `Copper`,
   `Plastic`, `Galvanized`, `Lead including lead-lined galvanized`, `Unknown`, **`Unknown but could be
   lead`**, `Unknown but unlikely lead`. The hedge is the load-bearing value — it is what NYC never
   uses and the rest of the state uses 15,285 times.
10. **`yearbuilt = 0` is MapPLUTO's unknown sentinel, not a year.** 4.6% of lots. Filter it or every
    era analysis grows a phantom pre-1800 bucket.

## Derived working files (`work/`, also uncommitted)

`scripts/extract.py` writes `work/t1_nyc_state.parquet` — one pass over the gz, all columns except
`Note`, 4,618,115 rows. Every later script reads that instead of re-parsing 99 MB of gzip.
`scripts/fetch_mappluto.py` writes `work/mappluto.json.gz` (858,602 NYC tax lots, `64uk-42ks`), and
`scripts/join_pluto.py` writes `work/nyc_joined.parquet` — 813,911 NYC addresses with construction
era joined from the nearest tax-lot centroid (median 23.4 m). Both are re-derivable from the two
committed provenance anchors plus a network call; neither is primary evidence.

`scripts/extract_t0.py` does the same for the archived 2025-06-22 capture (3,747,025 rows).
`scripts/statewide_screen.py`, `scripts/t0_t1.py`, `scripts/estimate.py`, `scripts/baseline2.py` and
`scripts/paper_numbers.py` read those parquets and write their results as JSON into `work/`, which is
what the figure scripts consume.

## Fetching the state's own documents

`health.ny.gov` returns 403 to a default user agent and 200 to a browser one:

    curl -L -A 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36' \
      -o guidance.pdf https://www.health.ny.gov/environmental/water/drinking/docs/service_line_inventory_guidance_lcrr.pdf

`osc.ny.gov` blocks even that; its PDFs come back from the Wayback Machine. A 403 from a state site is
a bot filter, not evidence the document is unavailable.

Out-of-bounds coordinates: at address level 343 NYC points (0.042% of those carrying a point) fall
outside a generous city bounding box. The extremes are in Oregon, Washington State, Maine and Florida.

## Verifying integrity

    cd ny-lsl-inventory && sha256sum -c SHA256SUMS

## If the `.gz` files are missing

T1 can be re-pulled with the `curl` above, though the numbers will have moved. **T0 probably cannot.**
Every quantity the archived snapshot supports is therefore written into the manuscript and into
`work/paper_numbers.json` rather than left implicit in the data.
