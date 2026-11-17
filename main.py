"""Forecast launch-day Twitch viewers for Grand Theft Auto VI, hour by hour.

Method: local-linear log-ratio regression over comparable past launches.

For every comparable launch we know its release-day hourly viewer counts, so we
can express its day as log-ratios to its own hour-1 print:

    r_h = ln(viewers_h / viewers_1)          for h = 2..24

A launch's log-ratio curve depends on how big the launch is (a small launch
rises steeply as regions wake up; a huge launch is already near its peak at
hour 1), so for each hour h separately we regress r_h on ln(viewers_1) across
the comparables, weighting each launch by a Gaussian kernel centred on the
target's own log hour-1 print:

    w_i = exp(-((ln(v1_i) - ln(p1)) / 1.2) ** 2)

The fitted value at the target's scale is the median log-ratio; the weighted
spread of the residuals around the fit gives the log-scale standard deviation,
and the five quantiles are the normal quantiles around the median, narrowed by
a factor of 0.85 (the raw residual spread is slightly over-wide on the four
practice launches).

Two filters keep the comparison honest:

* launches whose notes flag them "early-access-contaminated" or "weak" are
  dropped -- their pre-day curves were already running before release day, so
  their log-ratio shapes do not describe a clean midnight-local launch like
  GTA VI's;
* the Grand Theft Auto VI category's own pre-release history is not a launch
  and is dropped too.

Everything is deterministic: comparables are processed in name order, there is
no randomness, no network, no clock use, and only the standard library.

Fallbacks keep every case valid: an hour with too few comparable points falls
back to kernel-weighted empirical quantiles of the log-ratios, and an hour
with no points at all falls back to persistence (the hour-1 print with a
widening random-walk spread). If a case has no usable comparables at all, the
whole forecast is persistence.

Input:  every <id>.in in $MAKEFUN_INPUT_DIR (else /input).
Output: <id>.out in $MAKEFUN_OUTPUT_DIR (else /output):
        {"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}, ...]} for h = 2..24.
"""

import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SCORES = [-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722]

# Gaussian kernel width (in natural-log units of hour-1 viewers) for the
# scale weighting, and the factor applied to the residual spread.
KERNEL_SIGMA = 1.2
SPREAD_FACTOR = 0.85
# Fitted slopes are clamped so an extreme target scale cannot extrapolate
# the log-ratio trend without bound.
MAX_SLOPE = 1.5
# A comparable needs at least this many release-day hourly points to count.
MIN_DAY_POINTS = 15
# A kernel fit needs at least this many points for an hour; below that we use
# empirical quantiles of the log-ratios instead.
MIN_FIT_POINTS = 3
# Persistence fallback spread, in log units per square-root hour.
FALLBACK_SIGMA = 0.15

_TIER_RE = re.compile(r"Tier:\s*([a-z][a-z\-]*)")
_DROPPED_TIERS = {"early-access-contaminated", "weak", "target-history"}


def round_half_up_2(x):
    """Round to 2 decimals, half up; whole numbers come back as ints."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if float(value).is_integer() else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def day_hours(hourly, day_start):
    """Map release-day hour (1..24) -> viewers for one comparable launch."""
    hours = {}
    for row in hourly or []:
        viewers = row.get("viewers")
        if not viewers or viewers <= 0:
            continue
        try:
            at = parse_time(row["from"])
        except (KeyError, TypeError, ValueError):
            continue
        offset = (at - day_start).total_seconds()
        hour = int(offset // 3600) + 1
        if 1 <= hour <= 24:
            hours[hour] = viewers
    return hours


def launch_tier(comp):
    """The tier tag in a comparable's unlock note, if it carries one."""
    match = _TIER_RE.search(comp.get("unlock") or "")
    return match.group(1) if match else "comparable"


def collect_profiles(case):
    """Usable comparable launches: (log hour-1 viewers, {hour: log-ratio})."""
    profiles = []
    for comp in sorted(case.get("comparables") or [],
                       key=lambda c: c.get("name") or ""):
        name = comp.get("name") or ""
        if "Grand Theft Auto VI" in name:
            continue
        if launch_tier(comp) in _DROPPED_TIERS:
            continue
        try:
            day_start = parse_time(comp["day_start"])
        except (KeyError, TypeError, ValueError):
            continue
        hours = day_hours(comp.get("hourly"), day_start)
        if 1 not in hours or len(hours) < MIN_DAY_POINTS:
            continue
        first = hours[1]
        profiles.append((
            math.log(first),
            {h: math.log(v / first) for h, v in hours.items()},
        ))
    return profiles


def weighted_quantile(pairs, level):
    """Weighted empirical quantile of (weight, value) pairs."""
    ordered = sorted(pairs, key=lambda pair: pair[1])
    total = sum(w for w, _ in ordered)
    acc = 0.0
    for weight, value in ordered:
        acc += weight
        if acc >= level * total:
            return value
    return ordered[-1][1]


def local_linear_quantiles(profiles, log_p1, hour):
    """Five log-ratio quantiles for one hour, or None if no data at all."""
    points = []
    for log_first, ratios in profiles:
        if hour in ratios:
            weight = math.exp(-((log_first - log_p1) / KERNEL_SIGMA) ** 2)
            points.append((weight, log_first - log_p1, ratios[hour]))
    if not points:
        return None
    if len(points) < MIN_FIT_POINTS:
        pairs = [(w, y) for w, _, y in points]
        return [weighted_quantile(pairs, level) for level in LEVELS]

    total_w = sum(w for w, _, _ in points)
    mean_x = sum(w * x for w, x, _ in points) / total_w
    mean_y = sum(w * y for w, _, y in points) / total_w
    sxx = sum(w * (x - mean_x) ** 2 for w, x, _ in points)
    sxy = sum(w * (x - mean_x) * (y - mean_y) for w, x, y in points)
    slope = sxy / sxx if sxx > 1e-12 else 0.0
    slope = max(-MAX_SLOPE, min(MAX_SLOPE, slope))
    median = mean_y - slope * mean_x  # fitted value at x = 0 (target scale)
    variance = sum(w * (y - (median + slope * x)) ** 2
                   for w, x, y in points) / total_w
    spread = math.sqrt(max(variance, 0.0)) * SPREAD_FACTOR
    return [median + z * spread for z in Z_SCORES]


def persistence_quantiles(p1, hour):
    """Fallback: hour-1 print with a widening random-walk spread."""
    width = FALLBACK_SIGMA * math.sqrt(hour - 1)
    return [z * width for z in Z_SCORES]


def forecast_case(case):
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1),
                 None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    log_p1 = math.log(p1)
    profiles = collect_profiles(case)

    rows = []
    for row in sorted(case["forecast"], key=lambda r: r["hour"]):
        hour = row["hour"]
        log_q = local_linear_quantiles(profiles, log_p1, hour)
        if log_q is None:
            log_q = persistence_quantiles(p1, hour)
        quantiles = []
        for value in log_q:
            q = p1 * math.exp(value)
            q = max(1.0, min(1_000_000_000.0, q))
            quantiles.append(round_half_up_2(q))
        quantiles.sort()
        rows.append({"hour": hour, "q": quantiles})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out_path = OUTPUT_DIR / f"{path.stem}.out"
        out_path.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
