# GTA VI Twitch CCU Forecaster

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](main.py)
[![make.fun bounty](https://img.shields.io/badge/make.fun-bounty-7c3aed.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

Forecasts hour-by-hour Twitch concurrent viewers in the **Grand Theft Auto VI**
category on launch day, 19 November 2026 (UTC), hours 2–24, as five quantiles
(5 / 25 / 50 / 75 / 95%) — an entry for the
[make.fun bounty](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
($250, top 20).

## Method

`main.py` (Python 3.13, standard library only) reads one case file per run and
forecasts from the case's own data alone — no network, no hardcoded answers.

**Local-linear log-ratio regression over comparable launches.** Each past big
launch is expressed as log-ratios to its own hour-1 print,
`r_h = ln(viewers_h / viewers_1)`. For each hour separately, `r_h` is regressed
on `ln(viewers_1)` across the comparables, weighting each launch by a Gaussian
kernel centred on the target's own log hour-1 print (σ = 1.2 log units) — so
launches of a similar size matter most. The fitted value at the target's scale
is the median log-ratio; the weighted residual spread gives the log-scale
standard deviation, and the five quantiles are normal quantiles around the
median, narrowed by a factor of 0.85.

Two filters keep the comparison honest:

- Launches flagged `early-access-contaminated` or `weak` in their notes are
  dropped — their pre-day curves were already running before release day.
- The GTA VI category's own pre-release history is not a launch and is dropped.

Fallbacks keep every case valid: an hour with too few comparable points uses
kernel-weighted empirical quantiles of the log-ratios, and an hour with no
points at all falls back to persistence (hour-1 print, widening random-walk
spread). Everything is deterministic — no randomness, no clock, comparables
processed in name order.

## Local dev score (practice cases p00–p03)

Ground truth for the practice answers was taken from the same public source
the kit documents (SullyGnome hourly category averages, via the chart API
named in the kit's `input/NOTICE.txt`), scored with the exact pinball loss
from the rules (mean over every scored hour and quantile, log scale):

| case | launch | loss |
| ---- | ------ | ---- |
| p00 | Marvel's Wolverine | 0.193 |
| p01 | Red Dead Redemption 2 | 0.140 |
| p02 | God of War Ragnarök | 0.108 |
| p03 | Marvel's Spider-Man 2 | 0.198 |
| **mean** | | **0.160** |

For comparison, the house baseline (persistence, starter program) scores
**0.387** on the same four cases, and the best entry on the board at the time
of building scored **0.191**. Runtime is ~0.04 s per case; two runs produce
byte-identical output.

Practice scores are no guide to the final — the final case is a single new
launch day — but the method beats the baseline on every practice case.

## Run locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Files

- `main.py` — the entrypoint (stdlib only)
- `LICENSE` — MIT
- `.makefun/owner.json` — payout wallet

## License

MIT — see [LICENSE](LICENSE).
