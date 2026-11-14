# Polymarket FAA Attribution Forecaster

Entry for the make.fun `polymarket-faa-attribution-01` competition.
Forecasts Yes prices of 4 Polymarket FAA outage attribution markets at
minutes at+1…at+240 after each checkpoint, 5 quantiles
(0.05, 0.25, 0.5, 0.75, 0.95) per market-minute.

## Model

Persistence median + pooled empirical move quantiles:

- Median forecast is the market's last price (no drift).
- Move quantiles per horizon are estimated from the pooled price moves of
  every market in the case over the last 24 hours, at lags
  1, 5, 10, 20, 40, 80, 160 and 240 minutes, sampled every 5 minutes,
  interpolated log-linearly in the horizon.
- Moves are shrunk by 0.5 toward zero.

## Score

Local practice score (p00–p04 vs next case's prices): **0.336 pinball**
vs **0.393** for the house baseline. Official practice score on the
sibling competition (-02, same model): **0.295**.

## Run

```sh
python3 main.py   # reads /input/*.in, writes /output/*.out
```

Python 3.13, standard library only, offline, deterministic.
Quantile rows are clamped to [0, 1] and rounded to 6 decimals.
