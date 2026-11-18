# Security review — 20-point pen-test checklist (2026-11-19)

DriftWatch is a fully static, client-only ML-drift-monitoring demo — no
backend, no accounts, no database, and (unlike most other apps in this
portfolio) **no network calls at all**: all model/feature data is generated
from seeded Gaussians in-browser (`mulberry32` + Box–Muller), and PSI/KS
drift statistics are computed live from those samples. Reviewed `app.js`
and `core.js` in full; confirmed no `fetch(` calls anywhere in the repo.

| # | Check | Status | Notes |
|---|-------|--------|-------|
| 1 | HTTP/2 Rapid Reset | N/A | Static site, no backend server, no network calls of any kind. |
| 2 | bcrypt 72-byte limit | N/A | No passwords/accounts. |
| 3 | SAML signature bypass | N/A | No SAML. |
| 4 | nOAuth | N/A | No OAuth. |
| 5 | GraphQL alias batching | N/A | No GraphQL. |
| 6 | TOTP replay | N/A | No TOTP/2FA. |
| 7 | ReDoS | No | The only regex is `esc()`'s `/[&<>]/g` — simple, bounded. |
| 8 | JWT algorithm confusion | N/A | No JWTs. |
| 9 | HTTP request smuggling | N/A | No backend, no network calls at all. |
| 10 | Web cache poisoning | N/A | No backend responses. |
| 11 | Spoofed X-Forwarded-For | N/A | No server-side code, no IP-based logic. |
| 12 | Postgres views bypassing RLS | N/A | No database. |
| 13 | `Math.random()` for security tokens | N/A | The seeded `mulberry32` PRNG is deliberately deterministic (for reproducible demo statistics), not a security token, and there's no auth/session anywhere in the app. |
| 14 | Timing attacks on secret comparison | N/A | No secret comparison. |
| 15 | Padding oracle | N/A | No encryption. |
| 16 | Server-side prototype pollution | N/A | No backend at all. |
| 17 | Server-side template injection | N/A | No server-side templating. The only free-text input is the model search box (`monitors.html`'s `#q`), which is used purely as a substring filter against the app's own synthetic model-name list — not rendered back as HTML anywhere beyond the already-escaped model names it filters. |
| 18 | XXE | N/A | No XML parsing. |
| 19 | Cross-site WebSocket hijacking | N/A | No WebSocket endpoint. |
| 20 | Unicode username normalization | N/A | No usernames/login. |

## Summary

No findings. This is the lowest-risk app reviewed in this portfolio so
far — fully static, fully offline, with no network calls whatsoever (not
even a bring-your-own-key AI call) and no user-entered text that is ever
reflected back as HTML.
