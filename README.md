# DriftWatch — know when your model rots

Production ML models don't crash when they fail — they **quietly rot** as live data drifts away from what they were trained on. DriftWatch measures that drift with **real statistics** (Population Stability Index + Kolmogorov–Smirnov distance), tracks performance decay, and alerts you before it costs revenue. Everything runs **100% in the browser** — no data leaves your device.

## Run it
Open `index.html` in any modern browser, or serve the folder:
```bash
python3 -m http.server 8080   # then open http://localhost:8080
```
No build step, no dependencies, works fully offline.

## Pages
- **Home** — the pitch + a live 3D distribution-field hero.
- **Monitors** — every model ranked by drift; search + sort; health summary.
- **Model detail** — per-feature PSI & KS, 30-day drift history (with alert threshold), performance decay, prediction volume, and a baseline-vs-current distribution comparison you can switch per feature.
- **Alerts** — drag a PSI threshold and watch alerts recompute live across every model and feature, with severity.
- **Playground** — shift a distribution's mean and variance and watch PSI & KS move in real time. This is the exact math behind every monitor.
- **Settings** — theme, accents, private JSON export.

## The statistics are real
- **PSI** (`psi()`) bins baseline vs current into a shared range and sums `(aᵢ−eᵢ)·ln(aᵢ/eᵢ)` — the standard drift index (`<0.1` stable, `0.1–0.25` warning, `≥0.25` drift).
- **KS** (`ksStat()`) is the max gap between the two empirical CDFs.
- Model/feature data is generated from seeded Gaussians (`mulberry32` + Box–Muller) so numbers are stable across reloads, then PSI/KS are computed live from the samples — nothing is hard-coded.

## Deploy
Static site — drop the folder on Netlify (app.netlify.com/drop), `vercel`, or GitHub Pages. No env vars.

## Feature count
50+ — monitoring, PSI, KS, per-feature breakdown, drift history, decay tracking, volume trends, threshold alerting, a live playground, charts, 3D hero, theming, export, full offline. Full list on the home page.

*The real market need: silent model degradation is the #1 production-ML failure mode (cf. Arize, Fiddler, WhyLabs). DriftWatch is a self-contained, offline demonstration of how you catch it.*
