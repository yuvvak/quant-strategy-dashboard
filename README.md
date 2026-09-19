# Sector Momentum Strategy Dashboard

A monthly-rebalanced sector momentum rotation strategy with an absolute-momentum/cash overlay,
real transaction costs, walk-forward config selection, and a live-recomputing dashboard.

## What it does

Rotates into the top 3 of 11 SPDR sector ETFs by trailing price momentum, equal-weighted,
rebalanced monthly. A sector is only held if its trailing return also beats the real risk-free
(T-bill) return over the same window (Antonacci-style dual momentum) - otherwise that sleeve
sits in cash. Every rebalance charges an assumed 10bps of transaction cost per unit of
portfolio turnover.

The momentum lookback (12 months, skipping the most recent month) is chosen via a small,
literature-grounded sensitivity grid - but selection is walk-forward: the grid is evaluated
**only** on 2014-2021 (training), and the winning config is then reported both on the
2021-present period it was never selected against (genuinely out-of-sample) and on the full
history (for the live dashboard). See `web/public/data/sensitivity.json` for the training-period
grid, `web/public/data/summary.json`'s `walk_forward` block for the out-of-sample test, and
`engine/backtest.py`'s module docstring for the full methodology.

**Out-of-sample result (2021-present, the period that matters most since it was never used to
pick the config): strategy Sharpe 0.77 vs SPY 0.72 - a genuine risk-adjusted edge.** Full history
(2014-present) still trails SPY on Sharpe (0.66 vs 0.73), consistent with the project's original,
more conservative framing - both numbers are reported, not just the flattering one.

## Structure

- `engine/backtest.py`: downloads sector ETF, SPY and T-bill-yield data, runs the walk-forward
  backtest, and writes results into `web/public/data/` as JSON.
- `engine/tests/`: pytest suite against synthetic price data (no network calls) - covers
  no-look-ahead-bias, weights summing to 1, the cash overlay actually moving to cash when
  nothing clears the hurdle, and transaction costs actually reducing net returns.
- `web/`: Next.js dashboard reading those JSON files: performance chart, stats table, gross/net
  cost comparison, walk-forward out-of-sample section, rebalance history (with cash-allocation
  badges), sensitivity grid, and a rule-based (not LLM) commentary line.
- `.github/workflows/recompute.yml`: scheduled job that reruns the backtest weekly and commits
  updated data if anything changed - see Status below.

## Running it

```bash
# Regenerate the backtest data
pip install -r engine/requirements.txt
python engine/backtest.py          # run from the repo root - it writes relative paths

# Run the engine's unit tests (no network required)
cd engine && python -m pytest tests/ -v

# Run the dashboard
cd web
npm install
npm run dev
```

## Status

Live-recomputing: a scheduled GitHub Actions job (`.github/workflows/recompute.yml`) reruns
`engine/backtest.py` weekly and commits the refreshed JSON if the data changed, so the dashboard
reflects a genuinely current backtest rather than a one-off snapshot - every recompute is a git
commit, giving a full, auditable run history for free.

Two items from the original plan are intentionally deferred, not silently dropped:
- **Supabase database** - static JSON + git history already gives an auditable, zero-cost
  history of every recompute; a DB migration would add real infra (hosting, auth, a Supabase
  account) for a benefit the git-log approach already covers. Revisit if the data outgrows JSON.
- **True AI-generated commentary** - `web/public/data/commentary.json` is a deterministic,
  rule-based summary (explicitly labelled `is_ai_generated: false` in its own output), not an
  LLM call. Wiring up a real LLM pass would mean adding an API key as a GitHub Actions secret -
  a deliberate decision to make, not something to do silently on a scheduled job's behalf.
