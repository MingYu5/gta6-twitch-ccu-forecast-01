"""Forecast a launch day's Twitch viewers, hour by hour, from its first hour.

Method (fitted offline by leave-one-out over the comparable launches that
ship inside every case file; v2 constants re-tuned on the same LOO, scored
on the large launches the final resembles):

  * Each comparable with a complete release day becomes a log-ratio curve
    r(h) = ln(viewers(h) / viewers(hour 1)), h = 2..24.
  * Comparables are weighted by how much they resemble this case:
    a Gaussian kernel on the log of the hour-1 print (bandwidth 0.75) and on
    the pre-release ramp (mean log viewers of the six hours before the day
    starts, relative to hour 1; bandwidth 2.0), times 8 when the comparable's
    unlock pattern (rolling local midnights vs one simultaneous global
    unlock, read from its unlock note) matches this case's.
  * The forecast for an hour blends two views of the pool, 70/30: the
    weighted empirical quantiles of the pool's log-ratios at that hour
    (interpolated weighted quantiles), and the quantiles of the residuals
    around a per-hour weighted linear fit of log-ratio on log hour-1 size,
    which lets the shape trend extrapolate to launches larger than any in
    the pool. Both views shrink the inner quantiles 15% and the outer
    quantiles 10% toward their median; the blend is smoothed across
    neighbouring hours and pulled 10% toward the persistence shape.
  * With no usable comparable the program falls back to the persistence
    shape: every hour at hour 1's print, spread 0.15 * sqrt(h - 1).

The program is deterministic, offline and standard-library only.
It reads every <id>.in in the input directory and writes <id>.out with
{"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}, ...]}.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
SIGMA = 0.15

# Cross-validated constants (leave-one-out over the kit's comparables).
SCALE_BANDWIDTH = 0.75   # kernel width on ln(hour-1 viewers)
RAMP_BANDWIDTH = 2.0     # kernel width on the pre-release ramp feature
CLASS_MULTIPLIER = 8.0   # extra weight when the unlock pattern matches
INNER_SHRINK = 0.85      # 25%/75% quantiles toward the median
OUTER_SHRINK = 0.90      # 5%/95% quantiles toward the median
REG_BLEND = 0.3          # weight of the size-regression view in the blend
REG_RIDGE = 0.5          # ridge on the regression's weighted Sxx
BASE_BLEND = 0.1         # pull toward the persistence shape, in log space


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def round2(x):
    """Round to 2 decimals, half up; whole numbers come back as int."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if float(value).is_integer() else value


def unlock_class(text):
    """'rolling' for rolling local-midnight unlocks, 'global' for one
    simultaneous unlock, 'mixed' when a note names both, else 'unknown'."""
    t = (text or "").lower()
    rolling = ("midnight local" in t or "rolling" in t
               or "local time" in t or "each timezone" in t)
    global_ = ("simultaneous" in t or "single global" in t
               or "global unlock" in t)
    if rolling and not global_:
        return "rolling"
    if global_ and not rolling:
        return "global"
    if rolling or global_:
        return "mixed"
    return "unknown"


def hourly_by_offset(rows, day_start):
    """Map whole-hour offset from day_start to a row, for rows that carry
    a 'from' timestamp and a positive viewer count."""
    by = {}
    for row in rows or []:
        stamp = row.get("from") or row.get("at")
        if not stamp or not row.get("viewers"):
            continue
        off = (parse_ts(stamp) - day_start).total_seconds() / 3600.0
        if off == int(off):
            by[int(off)] = row
    return by


def ramp_feature(by, lv0):
    """Log of the mean viewers over the six hours before the day, minus
    ln(v0). Widens the window when the feed is sparse; 0.0 if unknown."""
    for span in (6, 24):
        vals = [by[o]["viewers"] for o in range(-span, 0) if o in by]
        if len(vals) >= 2:
            return math.log(sum(vals) / len(vals)) - lv0
    return 0.0


def weighted_quantile(pairs, tau):
    """Weighted quantile with linear interpolation on the midpoint CDF:
    each value sits at (cumweight - weight/2) / total, and tau between two
    midpoints interpolates linearly."""
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    pts = []
    acc = 0.0
    for value, w in pairs:
        pts.append((value, (acc + 0.5 * w) / total))
        acc += w
    if tau <= pts[0][1]:
        return pts[0][0]
    if tau >= pts[-1][1]:
        return pts[-1][0]
    for i in range(1, len(pts)):
        if pts[i][1] >= tau:
            v0, p0 = pts[i - 1]
            v1, p1 = pts[i]
            if p1 == p0:
                return v1
            return v0 + (v1 - v0) * (tau - p0) / (p1 - p0)
    return pts[-1][0]


def pool_curves(case, day_start):
    """(log-ratio curve, ln v0, ramp, unlock class) per usable comparable."""
    pool = []
    for comp in case.get("comparables") or []:
        try:
            start = parse_ts(comp["day_start"])
            by = hourly_by_offset(comp.get("hourly"), start)
        except (KeyError, ValueError):
            continue
        day = [by.get(h, {}).get("viewers") for h in range(24)]
        if any(v is None or v <= 0 for v in day):
            continue  # incomplete release day: not a usable curve
        lv0 = math.log(day[0])
        curve = [math.log(day[h] / day[0]) for h in range(1, 24)]
        pool.append((curve, lv0, ramp_feature(by, lv0),
                     unlock_class(comp.get("unlock"))))
    return pool


def forecast_log_quantiles(case):
    """Five log-quantiles relative to ln(hour-1 viewers), hours 2..24."""
    day_start = parse_ts(case["day_start"])
    first = next(k for k in case["known"] if k["hour"] == 1)
    p1 = max(1, first["viewers"])
    lv0 = math.log(p1)
    pool = pool_curves(case, day_start)
    if not pool:
        return [[z * SIGMA * math.sqrt(h - 1) for z in Z] for h in range(2, 25)]

    history = hourly_by_offset(case.get("history_hourly"), day_start)
    ramp = ramp_feature(history, lv0)
    cls = unlock_class(case.get("notes"))

    weights = []
    for _, clv0, cramp, ccls in pool:
        dist = ((clv0 - lv0) / SCALE_BANDWIDTH) ** 2 \
            + ((cramp - ramp) / RAMP_BANDWIDTH) ** 2
        w = math.exp(-0.5 * dist)
        if ccls == cls:
            w *= CLASS_MULTIPLIER
        weights.append(w)

    def shrunk_row(pairs, center):
        row = []
        for qi, tau in enumerate(LEVELS):
            q = weighted_quantile(list(pairs), tau)
            if qi in (0, 4):
                q = center + OUTER_SHRINK * (q - center)
            elif qi in (1, 3):
                q = center + INNER_SHRINK * (q - center)
            row.append(q)
        return row

    raw = []
    for hi in range(23):
        pairs = [(pool[j][0][hi], weights[j]) for j in range(len(pool))]
        median = weighted_quantile(list(pairs), 0.5)
        row = shrunk_row(pairs, median)
        if REG_BLEND:
            # Second view: weighted linear fit of this hour's log-ratio on
            # ln(hour-1 viewers); quantiles of its residuals, placed at the
            # fitted value for this case's size.
            sw = sum(weights)
            xm = sum(w * pool[j][1] for j, w in enumerate(weights)) / sw
            ym = sum(w * pool[j][0][hi] for j, w in enumerate(weights)) / sw
            sxx = sum(w * (pool[j][1] - xm) ** 2
                      for j, w in enumerate(weights))
            sxy = sum(w * (pool[j][1] - xm) * (pool[j][0][hi] - ym)
                      for j, w in enumerate(weights))
            slope = sxy / (sxx + REG_RIDGE)
            med_reg = ym + slope * (lv0 - xm)
            resid = [(pool[j][0][hi] - (ym + slope * (pool[j][1] - xm)),
                      weights[j]) for j in range(len(pool))]
            rrow = shrunk_row(resid, 0.0)
            rrow = [med_reg + v for v in rrow]
            row = [(1.0 - REG_BLEND) * a + REG_BLEND * b
                   for a, b in zip(row, rrow)]
        raw.append(row)

    # Smooth each quantile track across neighbouring hours, then re-anchor
    # every hour on its own median so the tracks cannot cross.
    smoothed = []
    for hi in range(23):
        lo, hi2 = max(0, hi - 1), min(23, hi + 2)
        row = [sum(raw[k][q] for k in range(lo, hi2)) / (hi2 - lo)
               for q in range(5)]
        smoothed.append(row)
    if BASE_BLEND:
        for hi in range(23):
            base = [z * SIGMA * math.sqrt(hi + 1) for z in Z]
            smoothed[hi] = [(1.0 - BASE_BLEND) * q + BASE_BLEND * b
                            for q, b in zip(smoothed[hi], base)]
    return smoothed


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    log_qs = forecast_log_quantiles(case)
    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        qs = [round2(p1 * math.exp(v)) for v in log_qs[h - 2]]
        qs = [min(10 ** 9, max(q, 1)) for q in qs]
        qs.sort()
        hours.append({"hour": h, "q": qs})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
