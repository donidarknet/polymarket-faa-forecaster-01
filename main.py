#!/usr/bin/env python3
"""Forecast Polymarket FAA outage attribution prices, minute by minute.

Model: persistence with empirical quantile bands.
- The median forecast is the market's last price (no drift: a 60-minute
  trend drift tested on the practice cases made the pinball loss far
  worse, 0.62 vs 0.39 for the baseline).
- For each horizon minute h, the move distribution is estimated from
  the pooled price moves of every market in the case over the last
  24 hours, at lags 1, 5, 10, 20, 40, 80, 160 and 240 minutes, sampled
  every 5 minutes, interpolated log-linearly in h.
- Moves are shrunk by 0.5: the raw empirical quantiles are too wide,
  because most minutes do not move at all and the occasional jump
  reverts quickly. Shrink 0.5 scored 0.336 on the practice cases
  (p00-p04, scored against the next case's prices) vs 0.393 for the
  house baseline on the same cases; 0.4 scored 0.336 too, but 0.5
  keeps wider bands as insurance against the larger jumps a
  news-driven final (the FAA markets) is more likely to see.
- Standard library only, no network, deterministic: the same case
  always writes the same bytes.

Input:  /input/<id>.in (or $MAKEFUN_INPUT_DIR), one JSON case:
  {"schema": "makefun/polymarket-case/v1", "case": <id>, "at": ...,
   "horizon_minutes": 240, "quantiles": [0.05, 0.25, 0.5, 0.75, 0.95],
   "markets": [{"id", "name", "slug", "yes_token", "start",
                "prices": [one per minute from start to at]}]}
Output: /output/<id>.out (or $MAKEFUN_OUTPUT_DIR), one JSON object:
  {"forecasts": [{"id": <market id>, "q": [[q05, q25, q50, q75, q95]
   x 240 rows]}]}
"""

import json
import math
import os
from pathlib import Path

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]
LAGS = [1, 5, 10, 20, 40, 80, 160, 240]
WINDOW_MINUTES = 1440  # last 24 hours of history per market
SAMPLE_STEP = 5        # sample move windows every 5 minutes
SHRINK = 0.5           # shrink empirical move quantiles toward 0


def _quantile(sorted_values, level):
    """Linear-interpolated quantile of a sorted list (level in [0, 1])."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_values[0]
    pos = level * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def pooled_move_quantiles(markets):
    """For each lag, the pooled move quantiles across all markets.

    Returns {lag: [q05, q25, q50, q75, q95] of the price move over
    that lag}, estimated from the last WINDOW_MINUTES of each market.
    """
    out = {}
    for lag in LAGS:
        moves = []
        for market in markets:
            prices = market["prices"][-WINDOW_MINUTES:]
            if len(prices) <= lag:
                continue
            for i in range(0, len(prices) - lag, SAMPLE_STEP):
                moves.append(prices[i + lag] - prices[i])
        if not moves:
            moves = [0.0]
        moves.sort()
        out[lag] = [_quantile(moves, level) for level in QUANTILES]
    return out


def _interpolated_quantiles(move_quantiles, horizon):
    """Move quantiles at minute `horizon`, log-linear in the lag."""
    if horizon <= LAGS[0]:
        return move_quantiles[LAGS[0]]
    if horizon >= LAGS[-1]:
        return move_quantiles[LAGS[-1]]
    for a, b in zip(LAGS, LAGS[1:]):
        if a <= horizon <= b:
            w = (math.log(horizon) - math.log(a)) / (math.log(b) - math.log(a))
            qa, qb = move_quantiles[a], move_quantiles[b]
            return [qa[i] * (1.0 - w) + qb[i] * w for i in range(len(QUANTILES))]
    return move_quantiles[LAGS[-1]]


def _round6(value):
    """Clamp to [0, 1] and round to 6 decimals (0, never -0)."""
    return round(min(1.0, max(0.0, value)) * 1e6) / 1e6 + 0.0


def forecast_market(last_price, move_quantiles, horizon_minutes):
    """240 rows of five quantiles around `last_price`."""
    rows = []
    for h in range(1, horizon_minutes + 1):
        moves = _interpolated_quantiles(move_quantiles, h)
        row = [_round6(last_price + SHRINK * move) for move in moves]
        # Quantiles must be non-decreasing.
        for i in range(1, len(row)):
            if row[i] < row[i - 1]:
                row[i] = row[i - 1]
        rows.append(row)
    return rows


def main():
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    case_files = sorted(input_dir.glob("*.in"))
    for case_file in case_files:
        case_id = case_file.name[: -len(".in")]
        with open(case_file, "r", encoding="utf-8") as fh:
            case = json.load(fh)
        markets = case["markets"]
        horizon = int(case["horizon_minutes"])
        move_quantiles = pooled_move_quantiles(markets)
        forecasts = []
        for market in markets:
            prices = market["prices"]
            if not prices:
                raise ValueError(f"market {market['id']} has no prices")
            last_price = prices[-1]
            forecasts.append(
                {
                    "id": market["id"],
                    "q": forecast_market(last_price, move_quantiles, horizon),
                }
            )
        out_path = output_dir / f"{case_id}.out"
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"forecasts": forecasts}) + "\n")


if __name__ == "__main__":
    main()
