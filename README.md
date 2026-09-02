# Sector Momentum Strategy Dashboard

A monthly-rebalanced sector momentum rotation strategy, backtested against SPY and an
equal-weight sector benchmark, with a live dashboard.

## What it does

Rotates into the top 3 of 11 SPDR sector ETFs by trailing price momentum, equal-weighted,
rebalanced monthly. The momentum lookback (12 months, skipping the most recent month) was
chosen via a small, literature-grounded sensitivity grid rather than picked because it looked
best in hindsight. See `web/public/data/sensitivity.json` for the full grid and
`engine/backtest.py` for methodology.

## Structure

- `engine/backtest.py`: downloads sector ETF price data, runs the backtest across a small
  parameter grid, and writes results directly into `web/public/data/` as JSON.
- `web/`: Next.js dashboard reading those JSON files: performance chart, stats table,
  rebalance history, and the sensitivity grid for transparency.

## Running it

```bash
# Regenerate the backtest data
pip install -r engine/requirements.txt
python engine/backtest.py

# Run the dashboard
cd web
npm install
npm run dev
```

## Status

Currently a backtest, not a live-tracked strategy. Planned next steps are a scheduled job
(GitHub Actions) to recompute the latest rebalance periodically, a Supabase database to store
running results instead of static JSON, and an AI-generated commentary layer, so the dashboard
becomes a genuinely live, continuously updating page rather than a point-in-time backtest.
