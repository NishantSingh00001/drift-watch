# DriftWatch — Build Log

**Problem:** Deployed ML models degrade silently when production data drifts from training data — the industry's top production-ML failure mode (Arize/Fiddler/WhyLabs category).

**Approach:** A self-contained, offline monitoring console built on the shared core framework (core.css/core.js) with a bespoke teal "lab" palette and a custom 3D hero.

## Real statistics (not mocked)
- `psi()` — Population Stability Index over a shared binning of baseline vs current.
- `ksStat()` — Kolmogorov–Smirnov max-CDF-gap.
- `mulberry32` seeded RNG + Box–Muller `gauss()` generate stable per-feature samples; PSI/KS computed live from them.
- 6 seeded models × 8 features each, with 30-day drift/accuracy/volume histories that ramp realistically.

## Custom 3D hero
`mountDriftHero()` — a point+wireframe surface rendering a Gaussian that shifts its mean and widens its variance over time (literal data drift), vertex-colored teal→mint→cyan, cursor-reactive. Additive materials so it renders vividly under headless swiftshader.

## Charts
Hand-written canvas line chart (`lineChart`) with gradient area fill, grid, value axis and a dashed threshold line; canvas histogram overlay (`drawHist`) for baseline-vs-current; CSS-bar histogram in the playground.

## Pages
index, monitors, model (detail, `?m=<id>`), alerts (live threshold), playground (live PSI/KS), settings.

## Testing (headless Chromium, swiftshader)
- Landing hero renders in color; 4 stat tiles. ✓
- Monitors: 6 model cards; search + sort; summary stats. ✓
- Model detail: 8 feature rows, 4 canvas charts, histogram + feature selector. ✓
- Alerts: 21 alerts generated; threshold slider recomputes. ✓
- Playground: 22 histogram bars; live verdict (STABLE/WARNING/DRIFT); presets. ✓
- Settings: theme toggle, accents, export. ✓
- **0 JavaScript errors** across all pages.

## Status
Complete. 6 pages, 50+ features, real drift math, 3D hero, 0 errors. Delivered as driftwatch.zip.
