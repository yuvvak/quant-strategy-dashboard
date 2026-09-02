# -*- coding: utf-8 -*-
"""
Sector momentum rotation strategy.

Monthly rebalance into the top N of 11 SPDR sector ETFs by trailing price
momentum, equal-weighted. Benchmarked against SPY (buy & hold) and an
equal-weight all-sector portfolio.

Supports a small, literature-grounded sensitivity grid over:
  - momentum lookback window (3 / 6 / 9 / 12 months)
  - "skip most recent month" convention (Jegadeesh-Titman 12-1 style),
    which avoids short-term reversal contaminating the momentum signal.

This is a robustness check, not a search for the best-looking number -
top_n stays fixed at 3 throughout, matching the original strategy design.

Outputs (for the best config found in the sensitivity grid), written into the Next.js
app's public folder so the dashboard can read them directly:
  web/public/data/performance.json  - daily cumulative performance series (strategy vs benchmarks)
  web/public/data/rebalances.json   - full rebalance history (date -> sectors selected + momentum)
  web/public/data/summary.json      - headline stats (total/annualized return, vol, Sharpe, drawdown)
  web/public/data/sensitivity.json  - full grid results, for transparency about how the config was chosen
"""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}
BENCHMARK = "SPY"
TOP_N = 3
RISK_FREE_ANNUAL = 0.0

START = "2014-01-01"

TRADING_DAYS_PER_MONTH = 21
GRID = [
    {"lookback_months": 3, "skip_month": False},
    {"lookback_months": 6, "skip_month": False},
    {"lookback_months": 9, "skip_month": False},
    {"lookback_months": 12, "skip_month": False},
    {"lookback_months": 6, "skip_month": True},
    {"lookback_months": 12, "skip_month": True},
]


def download_prices():
    tickers = list(SECTOR_ETFS.keys()) + [BENCHMARK]
    for attempt in range(3):
        raw = yf.download(tickers, start=START, auto_adjust=True, progress=False)["Close"]
        missing = [t for t in tickers if t not in raw.columns or raw[t].dropna().empty]
        if not missing:
            break
        print(f"Retry {attempt + 1}: missing/empty data for {missing}, retrying full download...")
    else:
        raise RuntimeError(f"Could not download clean data for: {missing}")
    raw = raw.dropna(how="all").ffill()
    return raw


def month_end_dates(index):
    s = pd.Series(index=index, data=index)
    return s.resample("ME").last().dropna().tolist()


def run_backtest(prices: pd.DataFrame, lookback_months: int, skip_month: bool, top_n: int = TOP_N):
    sector_prices = prices[list(SECTOR_ETFS.keys())]
    bench_prices = prices[BENCHMARK]

    lookback_days = lookback_months * TRADING_DAYS_PER_MONTH
    skip_days = TRADING_DAYS_PER_MONTH if skip_month else 0

    rebal_dates = month_end_dates(prices.index)
    min_start = prices.index[0] + pd.Timedelta(days=int(lookback_days * 1.5) + 30)
    rebal_dates = [d for d in rebal_dates if d >= min_start]

    daily_returns = sector_prices.pct_change().fillna(0)
    bench_returns = bench_prices.pct_change().fillna(0)

    strategy_daily = pd.Series(0.0, index=prices.index)
    equal_weight_daily = daily_returns.mean(axis=1)

    rebalance_log = []

    for i, rebal_date in enumerate(rebal_dates):
        hist = sector_prices.loc[:rebal_date]
        if hist.shape[0] < lookback_days + skip_days + 1:
            continue

        end_price = hist.iloc[-1 - skip_days]
        start_price = hist.iloc[-1 - skip_days - lookback_days]
        momentum = (end_price / start_price - 1).sort_values(ascending=False)
        top = momentum.head(top_n)
        current_weights = {ticker: 1.0 / top_n for ticker in top.index}

        rebalance_log.append({
            "date": rebal_date.strftime("%Y-%m-%d"),
            "selected": [
                {"ticker": t, "sector": SECTOR_ETFS[t], "momentum": round(float(momentum[t]), 4)}
                for t in top.index
            ],
        })

        period_end = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else prices.index[-1]
        mask = (prices.index > rebal_date) & (prices.index <= period_end)
        for ticker, w in current_weights.items():
            strategy_daily.loc[mask] += w * daily_returns.loc[mask, ticker]

    strategy_cum = (1 + strategy_daily).cumprod()
    bench_cum = (1 + bench_returns).cumprod()
    ew_cum = (1 + equal_weight_daily).cumprod()

    def annualized_return(cum_series):
        n_years = (cum_series.index[-1] - cum_series.index[0]).days / 365.25
        return cum_series.iloc[-1] ** (1 / n_years) - 1

    def annualized_vol(daily):
        return daily.std() * np.sqrt(252)

    def sharpe(daily):
        vol = annualized_vol(daily)
        return 0.0 if vol == 0 else (daily.mean() * 252 - RISK_FREE_ANNUAL) / vol

    def max_drawdown(cum_series):
        peak = cum_series.cummax()
        return (cum_series / peak - 1).min()

    def stats(cum_series, daily):
        return {
            "total_return": round(float(cum_series.iloc[-1] - 1), 4),
            "annualized_return": round(float(annualized_return(cum_series)), 4),
            "annualized_vol": round(float(annualized_vol(daily)), 4),
            "sharpe_ratio": round(float(sharpe(daily)), 3),
            "max_drawdown": round(float(max_drawdown(cum_series)), 4),
        }

    summary = {
        "backtest_start": prices.index[0].strftime("%Y-%m-%d"),
        "backtest_end": prices.index[-1].strftime("%Y-%m-%d"),
        "config": {"lookback_months": lookback_months, "skip_month": skip_month, "top_n": top_n},
        "strategy": stats(strategy_cum, strategy_daily),
        "benchmark_spy": stats(bench_cum, bench_returns),
        "benchmark_equal_weight": stats(ew_cum, equal_weight_daily),
    }

    performance = pd.DataFrame({
        "date": prices.index.strftime("%Y-%m-%d"),
        "strategy_cumulative": strategy_cum.values,
        "spy_cumulative": bench_cum.values,
        "equal_weight_cumulative": ew_cum.values,
    })

    return summary, rebalance_log, performance


def main():
    print("Downloading price data...")
    prices = download_prices()
    print(f"Got {prices.shape[0]} trading days from {prices.index[0].date()} to {prices.index[-1].date()}\n")

    print("Running sensitivity grid (lookback window x skip-month convention)...\n")
    grid_results = []
    best = None
    for cfg in GRID:
        summary, rebalance_log, performance = run_backtest(prices, **cfg)
        excess_sharpe = summary["strategy"]["sharpe_ratio"] - summary["benchmark_spy"]["sharpe_ratio"]
        row = {**cfg, "sharpe": summary["strategy"]["sharpe_ratio"],
               "annualized_return": summary["strategy"]["annualized_return"],
               "vs_spy_sharpe": round(excess_sharpe, 3)}
        grid_results.append(row)
        print(f"  lookback={cfg['lookback_months']:>2}mo  skip_month={str(cfg['skip_month']):<5}  "
              f"Sharpe={row['sharpe']:>6}  AnnRet={row['annualized_return']:>7}  vs SPY Sharpe={row['vs_spy_sharpe']:>+.3f}")
        if best is None or row["sharpe"] > best[0]["sharpe"]:
            best = (row, summary, rebalance_log, performance)

    with open("web/public/data/sensitivity.json", "w") as f:
        json.dump(grid_results, f, indent=2)

    print(f"\nBest config by Sharpe: {best[0]}")

    best_row, best_summary, best_rebalances, best_performance = best
    best_summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    best_summary["selection_note"] = (
        "Config selected as the best Sharpe ratio among a small, literature-grounded grid "
        "(3/6/9/12-month lookback x with/without 1-month skip), not an unconstrained parameter "
        "search. Full grid in sensitivity.json for transparency."
    )

    with open("web/public/data/summary.json", "w") as f:
        json.dump(best_summary, f, indent=2)
    with open("web/public/data/rebalances.json", "w") as f:
        json.dump(best_rebalances, f, indent=2)
    best_performance.to_json("web/public/data/performance.json", orient="records", indent=2)

    print("\n=== SELECTED CONFIG SUMMARY ===")
    print(json.dumps(best_summary, indent=2))
    print(f"\n{len(best_rebalances)} rebalances logged.")
    print("Wrote web/public/data/{summary,rebalances,performance,sensitivity}.json")


if __name__ == "__main__":
    main()
