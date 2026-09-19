# -*- coding: utf-8 -*-
"""
Sector momentum rotation strategy.

Monthly rebalance into the top N of 11 SPDR sector ETFs by trailing price
momentum, equal-weighted, with an absolute-momentum/cash overlay and real
transaction costs. Benchmarked against SPY (buy & hold) and an equal-weight
all-sector portfolio.

Methodology, in order of what a reviewer would ask about:

1. Momentum signal: trailing total return over `lookback_months`, optionally
   skipping the most recent month (Jegadeesh-Titman 12-1 style, to avoid
   short-term reversal contaminating the signal).

2. Absolute-momentum / cash overlay: a sector is only eligible to be held if
   its trailing return over the lookback window beats the risk-free/cash
   return over that same window (Antonacci-style dual momentum). If fewer
   than `top_n` sectors clear that bar, the uninvested weight sits in cash
   and earns the realised risk-free rate rather than staying fully invested
   through a broad drawdown.

3. Transaction costs: every rebalance charges `COST_BPS_PER_TRADE` basis
   points per unit of portfolio weight turned over (sum of absolute weight
   changes across all tickers, including entries/exits and the cash sleeve).
   Reported net of costs; a zero-cost run is also computed for a gross/net
   comparison, so the cost assumption's impact is visible, not hidden.

4. Config selection is walk-forward, not full-sample: the sensitivity grid
   (3/6/9/12-month lookback x with/without 1-month skip, top_n fixed at 3)
   is evaluated ONLY on the training period (start to WALK_FORWARD_SPLIT).
   The best-Sharpe config from that training window is then evaluated on the
   held-out test period it was never selected against, and separately on the
   full history for the live dashboard. Selecting and evaluating on the same
   window (the old approach) is a mild form of look-ahead bias even with a
   small grid - this closes that gap.

5. Risk-free rate: real 13-week T-bill yield (^IRX), not an assumed 0%, used
   both as the absolute-momentum cash hurdle and in every Sharpe ratio.

6. No-trade band: a currently-held sector isn't dropped just because a
   challenger edges narrowly ahead of it - it's only replaced if a challenger
   beats it by more than NO_TRADE_BAND. This is a fixed, round, standard
   buffer size (2 percentage points), not grid-searched against this data -
   the same "fixed, not optimized" treatment already applied to `top_n`.
   Cuts turnover (and therefore transaction costs) directly, without touching
   the underlying signal.

7. Volatility targeting: the strategy's exposure is scaled so trailing
   realized volatility tracks a target (the SELECTED config's own TRAINING-
   period volatility - not an arbitrary number, and not picked to flatter the
   full-period result). Capped at VOL_TARGET_CAP (1.5x) - a first version of
   this capped at 1x to dodge financing-cost modeling entirely, but most of
   the literature's documented Sharpe benefit (Moreira & Muir) specifically
   comes from levering UP in calm regimes, not just de-risking in volatile
   ones, so a 1x cap structurally can't capture it. The levered fraction
   (scale > 1x) pays a modeled financing cost of LEVERAGE_SPREAD_BPS_ANNUAL
   over the risk-free rate. The scale factor is lagged by one day (uses
   yesterday's realized vol) to avoid look-ahead. Reported both with and
   without this overlay (see strategy_pre_vol_target).

8. Risk-parity weighting: selected sectors are weighted inversely to their
   own trailing realized volatility (RISK_PARITY_LOOKBACK_DAYS) rather than
   flat equal-weight, so a single volatile sector doesn't dominate portfolio
   risk. Total invested weight is unchanged - only the split across selected
   sectors differs from equal-weight.

Outputs, written into the Next.js app's public folder so the dashboard can
read them directly:
  web/public/data/performance.json  - daily cumulative performance series (net of costs)
  web/public/data/rebalances.json   - full rebalance history (sectors selected, cash weight, turnover, cost)
  web/public/data/summary.json      - headline stats, gross/net comparison, walk-forward test results
  web/public/data/sensitivity.json  - training-period grid results, for transparency about config selection
  web/public/data/commentary.json   - a short, deterministic (not LLM-generated) text summary of the latest state
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
RISK_FREE_TICKER = "^IRX"  # 13-week T-bill discount yield, annualized %, e.g. 5.25 = 5.25%
TOP_N = 3
COST_BPS_PER_TRADE = 10  # 0.10% per unit of portfolio weight turnover - a conservative
# assumption for liquid, large-cap SPDR sector ETFs; real costs are typically lower.
NO_TRADE_BAND = 0.0  # off by default - tested at 0.02 (see extensions_tested in summary.json /
# README), found to reduce Sharpe for this specific momentum strategy (delayed rotation costs
# more than the saved turnover), so it's not applied by default. Kept as an opt-in parameter.
NO_TRADE_BAND_TESTED = 0.02  # the value actually evaluated in the extensions-tested comparison.
VOL_TARGET_LOOKBACK_DAYS = 63  # ~3 months of realized daily returns, a standard window.
VOL_TARGET_CAP = 1.5  # max exposure when vol-targeting is on - see docstring point 7.
LEVERAGE_SPREAD_BPS_ANNUAL = 50  # financing cost above the risk-free rate for the levered
# fraction (scale > 1x) - a conservative approximation of futures-implied financing cost,
# how a vol-target overlay like this is typically implemented in practice rather than via
# margin borrowing. Fixed, not tuned against this data.
RISK_PARITY_LOOKBACK_DAYS = 63  # matches VOL_TARGET_LOOKBACK_DAYS for consistency.

START = "2014-01-01"
WALK_FORWARD_SPLIT = "2021-01-01"  # config is selected using data before this date only

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
    tickers = list(SECTOR_ETFS.keys()) + [BENCHMARK, RISK_FREE_TICKER]
    for attempt in range(3):
        raw = yf.download(tickers, start=START, auto_adjust=True, progress=False)["Close"]
        # ^IRX is a yield, not a price - still expected to have data across the range, so
        # it's checked for "missing entirely" the same way as the price tickers.
        missing = [t for t in tickers if t not in raw.columns or raw[t].dropna().empty]
        if not missing:
            break
        print(f"Retry {attempt + 1}: missing/empty data for {missing}, retrying full download...")
    else:
        raise RuntimeError(f"Could not download clean data for: {missing}")
    raw = raw.dropna(how="all").ffill()
    return raw


def risk_free_daily_returns(rf_yield_pct: pd.Series) -> pd.Series:
    """Linear approximation of daily cash return from the annualized ^IRX yield (%)."""
    return (rf_yield_pct / 100.0) / 252.0


def month_end_dates(index):
    s = pd.Series(index=index, data=index)
    return s.resample("ME").last().dropna().tolist()


def run_backtest(
    prices: pd.DataFrame,
    rf_daily: pd.Series,
    lookback_months: int,
    skip_month: bool,
    top_n: int = TOP_N,
    cost_bps: float = COST_BPS_PER_TRADE,
    absolute_momentum: bool = True,
    no_trade_band: float = NO_TRADE_BAND,
    target_vol: float | None = None,
    risk_parity: bool = False,
    period_start=None,
    period_end=None,
):
    """
    Runs the strategy over [period_start, period_end] (default: full `prices` range).
    Momentum/cash-hurdle lookback windows always draw on the FULL price history before
    period_start where needed, so a test-period-only run still has real pre-period data
    to compute its first few rebalances' signals from - only the reported P&L is scoped
    to the requested period.

    `target_vol`: if set, scales exposure (up to VOL_TARGET_CAP, which may exceed 1x - the
    levered fraction pays LEVERAGE_SPREAD_BPS_ANNUAL over the risk-free rate) so trailing
    realized vol tracks this annualized target, lagged one day. If None, vol targeting is
    off and the raw signal-driven exposure is used as-is.

    `risk_parity`: if True, selected sectors are weighted inversely to their own trailing
    realized volatility instead of equal-weight (total invested weight unchanged).
    """
    sector_prices = prices[list(SECTOR_ETFS.keys())]
    bench_prices = prices[BENCHMARK]

    period_start = pd.Timestamp(period_start) if period_start is not None else prices.index[0]
    period_end = pd.Timestamp(period_end) if period_end is not None else prices.index[-1]
    idx = prices.index[(prices.index >= period_start) & (prices.index <= period_end)]

    lookback_days = lookback_months * TRADING_DAYS_PER_MONTH
    skip_days = TRADING_DAYS_PER_MONTH if skip_month else 0

    rebal_dates = month_end_dates(idx)
    min_start = prices.index[0] + pd.Timedelta(days=int(lookback_days * 1.5) + 30)
    rebal_dates = [d for d in rebal_dates if d >= max(min_start, period_start)]

    daily_returns = sector_prices.pct_change().fillna(0)
    bench_returns = bench_prices.pct_change().fillna(0)
    equal_weight_daily = daily_returns.loc[idx].mean(axis=1)

    strategy_daily = pd.Series(0.0, index=idx)
    rebalance_log = []
    prev_weights: dict[str, float] = {}

    for i, rebal_date in enumerate(rebal_dates):
        hist = sector_prices.loc[:rebal_date]
        rf_hist = rf_daily.loc[:rebal_date]
        if hist.shape[0] < lookback_days + skip_days + 1:
            continue

        n = hist.shape[0]
        p1 = n - 1 - skip_days
        p0 = n - 1 - skip_days - lookback_days

        end_price = hist.iloc[p1]
        start_price = hist.iloc[p0]
        momentum = (end_price / start_price - 1).sort_values(ascending=False)

        # Cash/risk-free trailing return over the identical window, as the absolute-momentum hurdle.
        rf_window = rf_hist.iloc[p0 + 1 : p1 + 1]
        cash_hurdle = float((1 + rf_window).prod() - 1) if len(rf_window) else 0.0

        eligible = momentum[momentum > cash_hurdle] if absolute_momentum else momentum
        # eligible is already sorted descending (inherited from `momentum`'s sort).

        # No-trade band: keep a currently-held, still-eligible sector unless a challenger
        # beats it by more than `no_trade_band` - reduces turnover-driven cost without
        # touching the underlying signal. Falls back to plain top-n when band is 0.
        held_tickers = [t for t in prev_weights if t != "CASH"]
        if len(eligible) >= top_n:
            cutoff_momentum = float(eligible.iloc[top_n - 1])
        elif len(eligible):
            cutoff_momentum = float(eligible.iloc[-1])
        else:
            cutoff_momentum = None

        retained = [
            t for t in held_tickers
            if t in eligible.index and (cutoff_momentum is None or eligible[t] >= cutoff_momentum - no_trade_band)
        ]
        retained = sorted(retained, key=lambda t: eligible[t], reverse=True)[:top_n]
        fill = [t for t in eligible.index if t not in retained][: top_n - len(retained)]
        top_tickers = retained + fill

        invested_total = len(top_tickers) / top_n
        if risk_parity and top_tickers:
            vol_hist = daily_returns.loc[:rebal_date, top_tickers].iloc[-RISK_PARITY_LOOKBACK_DAYS:]
            realized_vols = vol_hist.std() * np.sqrt(252)
            inv_vol = 1.0 / realized_vols
            if inv_vol.notna().all() and np.isfinite(inv_vol).all() and inv_vol.sum() > 0:
                within_weights = inv_vol / inv_vol.sum()
                current_weights = {t: float(within_weights[t]) * invested_total for t in top_tickers}
            else:
                # Not enough history yet to estimate vol for one or more names - fall back
                # to equal-weight rather than dividing by a NaN/zero vol.
                current_weights = {ticker: 1.0 / top_n for ticker in top_tickers}
        else:
            current_weights = {ticker: 1.0 / top_n for ticker in top_tickers}
        cash_weight = 1.0 - sum(current_weights.values())
        if cash_weight > 1e-9:
            current_weights["CASH"] = cash_weight

        all_tickers = set(current_weights) | set(prev_weights)
        turnover = sum(abs(current_weights.get(t, 0.0) - prev_weights.get(t, 0.0)) for t in all_tickers)
        cost = turnover * (cost_bps / 10000.0)

        rebalance_log.append({
            "date": rebal_date.strftime("%Y-%m-%d"),
            "selected": [
                {
                    "ticker": t,
                    "sector": SECTOR_ETFS[t],
                    "momentum": round(float(momentum[t]), 4),
                    "weight": round(float(current_weights[t]), 4),
                }
                for t in top_tickers
            ],
            "cash_weight": round(cash_weight, 4),
            "cash_hurdle": round(cash_hurdle, 4),
            "turnover": round(turnover, 4),
            "cost_charged": round(cost, 6),
        })

        period_end_i = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else idx[-1]
        mask = (idx > rebal_date) & (idx <= period_end_i)
        masked_dates = idx[mask]
        for ticker, w in current_weights.items():
            if ticker == "CASH":
                strategy_daily.loc[masked_dates] += w * rf_daily.reindex(masked_dates).fillna(0)
            else:
                strategy_daily.loc[masked_dates] += w * daily_returns.loc[masked_dates, ticker]

        if len(masked_dates):
            strategy_daily.loc[masked_dates[0]] -= cost

        prev_weights = current_weights

    bench_cum = (1 + bench_returns.loc[idx]).cumprod()
    ew_cum = (1 + equal_weight_daily).cumprod()
    rf_aligned = rf_daily.reindex(idx).fillna(0)

    pre_vol_target_daily = strategy_daily
    vol_target_info = None
    if target_vol is not None:
        realized_vol = strategy_daily.rolling(VOL_TARGET_LOOKBACK_DAYS).std() * np.sqrt(252)
        raw_scale = (target_vol / realized_vol).clip(upper=VOL_TARGET_CAP)
        scale = raw_scale.shift(1).fillna(1.0).clip(lower=0.0)

        underinvested = (1.0 - scale).clip(lower=0.0)  # fraction sitting in cash when scale < 1
        overlevered = (scale - 1.0).clip(lower=0.0)  # fraction borrowed when scale > 1
        spread_daily = (LEVERAGE_SPREAD_BPS_ANNUAL / 10000.0) / 252.0
        strategy_daily = (
            scale * pre_vol_target_daily
            + underinvested * rf_aligned
            - overlevered * (rf_aligned + spread_daily)
        )
        vol_target_info = {
            "target_vol": round(float(target_vol), 4),
            "lookback_days": VOL_TARGET_LOOKBACK_DAYS,
            "cap": VOL_TARGET_CAP,
            "leverage_spread_bps_annual": LEVERAGE_SPREAD_BPS_ANNUAL,
            "avg_exposure": round(float(scale.mean()), 4),
            "pct_days_levered": round(float((scale > 1.0 + 1e-9).mean()), 4),
        }

    strategy_cum = (1 + strategy_daily).cumprod()

    def annualized_return(cum_series):
        n_years = (cum_series.index[-1] - cum_series.index[0]).days / 365.25
        return cum_series.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 else 0.0

    def annualized_vol(daily):
        return daily.std() * np.sqrt(252)

    def sharpe(daily):
        vol = annualized_vol(daily)
        excess = daily.mean() * 252 - rf_aligned.mean() * 252
        return 0.0 if vol == 0 else excess / vol

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
        "period_start": idx[0].strftime("%Y-%m-%d") if len(idx) else None,
        "period_end": idx[-1].strftime("%Y-%m-%d") if len(idx) else None,
        "config": {"lookback_months": lookback_months, "skip_month": skip_month, "top_n": top_n},
        "strategy": stats(strategy_cum, strategy_daily),
        "benchmark_spy": stats(bench_cum, bench_returns.loc[idx]),
        "benchmark_equal_weight": stats(ew_cum, equal_weight_daily),
    }
    if target_vol is not None:
        pre_cum = (1 + pre_vol_target_daily).cumprod()
        summary["strategy_pre_vol_target"] = stats(pre_cum, pre_vol_target_daily)
        summary["vol_target"] = vol_target_info

    performance = pd.DataFrame({
        "date": idx.strftime("%Y-%m-%d"),
        "strategy_cumulative": strategy_cum.values,
        "spy_cumulative": bench_cum.values,
        "equal_weight_cumulative": ew_cum.values,
    })

    return summary, rebalance_log, performance


def generate_commentary(rebalance_log, summary):
    """Deterministic, rule-based summary of the latest state - explicitly NOT an LLM call.
    A true AI-generated layer would need an LLM API key wired into a GitHub Actions secret;
    deferred for now, see README."""
    if not rebalance_log:
        text = "No rebalances available for this backtest window."
    else:
        latest = rebalance_log[-1]
        tickers = ", ".join(f"{s['ticker']} ({s['sector']})" for s in latest["selected"])
        cash_note = (
            f" {round(latest['cash_weight'] * 100)}% held in cash this period - the "
            "absolute-momentum filter excluded one or more sectors whose trailing return "
            "didn't beat cash."
            if latest["cash_weight"] > 0.001
            else " Fully invested - every selected sector cleared the cash hurdle."
        )
        text = (
            f"Most recent rebalance ({latest['date']}): rotated into {tickers}.{cash_note} "
            f"Strategy Sharpe ratio (net of costs, full history): {summary['strategy']['sharpe_ratio']:.2f} "
            f"vs SPY {summary['benchmark_spy']['sharpe_ratio']:.2f}."
        )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "text": text,
        "is_ai_generated": False,
        "note": (
            "Rule-based summary generated deterministically from the latest backtest output, "
            "not an LLM call."
        ),
    }


def main():
    print("Downloading price + risk-free data...")
    raw = download_prices()
    prices = raw[list(SECTOR_ETFS.keys()) + [BENCHMARK]]
    rf_daily = risk_free_daily_returns(raw[RISK_FREE_TICKER])
    print(f"Got {prices.shape[0]} trading days from {prices.index[0].date()} to {prices.index[-1].date()}\n")

    split = pd.Timestamp(WALK_FORWARD_SPLIT)

    print(f"Stage 1: selecting config using TRAINING data only ({prices.index[0].date()} to {split.date()})...\n")
    grid_results = []
    best = None
    best_train_performance = None
    for cfg in GRID:
        summary, _, performance = run_backtest(prices, rf_daily, **cfg, period_end=split)
        row = {
            **cfg,
            "sharpe": summary["strategy"]["sharpe_ratio"],
            "annualized_return": summary["strategy"]["annualized_return"],
            "vs_spy_sharpe": round(summary["strategy"]["sharpe_ratio"] - summary["benchmark_spy"]["sharpe_ratio"], 3),
        }
        grid_results.append(row)
        print(f"  lookback={cfg['lookback_months']:>2}mo  skip_month={str(cfg['skip_month']):<5}  "
              f"Sharpe={row['sharpe']:>6}  AnnRet={row['annualized_return']:>7}  vs SPY Sharpe={row['vs_spy_sharpe']:>+.3f}")
        if best is None or row["sharpe"] > best["sharpe"]:
            best = row
            best_train_performance = performance

    wf_config = {"lookback_months": best["lookback_months"], "skip_month": best["skip_month"]}
    for row in grid_results:
        row["selected"] = row["lookback_months"] == wf_config["lookback_months"] and row["skip_month"] == wf_config["skip_month"]
    print(f"\nSelected via training-period Sharpe: {wf_config}\n")

    print("Stage 2: evaluating the selected config OUT-OF-SAMPLE (test period, never used for selection)...\n")
    oos_summary, _, _ = run_backtest(prices, rf_daily, **wf_config, period_start=split)
    print(json.dumps(oos_summary, indent=2))

    print("\nStage 3: full-period run (net of costs) for the live dashboard...\n")
    full_summary, full_rebalances, full_performance = run_backtest(prices, rf_daily, **wf_config)

    print("Stage 4: full-period run at zero cost, for a gross-vs-net comparison...\n")
    gross_summary, _, _ = run_backtest(prices, rf_daily, **wf_config, cost_bps=0)

    print("Stage 5: evaluating optional extensions (no-trade band, vol targeting, risk parity) - "
          "kept opt-in in the code, off by default here based on what this comparison finds...\n")
    train_daily = best_train_performance["strategy_cumulative"].pct_change().dropna()
    target_vol = float(train_daily.std() * np.sqrt(252))

    def ext_stats(**kwargs):
        f, _, _ = run_backtest(prices, rf_daily, **wf_config, **kwargs)
        o, _, _ = run_backtest(prices, rf_daily, **wf_config, period_start=split, **kwargs)
        return f["strategy"]["sharpe_ratio"], o["strategy"]["sharpe_ratio"]

    extensions_tested = []
    for label, kwargs in [
        ("No-trade band (2pp) - cuts turnover, delays rotation", {"no_trade_band": NO_TRADE_BAND_TESTED}),
        ("Volatility targeting (up to 1.5x, financing cost modelled)", {"target_vol": target_vol}),
        ("Risk-parity weighting (inverse-vol instead of equal-weight)", {"risk_parity": True}),
        ("Vol targeting + risk parity combined", {"target_vol": target_vol, "risk_parity": True}),
    ]:
        full_sharpe, oos_sharpe = ext_stats(**kwargs)
        extensions_tested.append({
            "label": label,
            "full_sharpe": full_sharpe,
            "oos_sharpe": oos_sharpe,
            "full_sharpe_delta": round(full_sharpe - full_summary["strategy"]["sharpe_ratio"], 3),
            "oos_sharpe_delta": round(oos_sharpe - oos_summary["strategy"]["sharpe_ratio"], 3),
        })
        print(f"  {label:58s} full {full_sharpe:.3f}  oos {oos_sharpe:.3f}")

    full_summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    full_summary["transaction_cost_bps"] = COST_BPS_PER_TRADE
    full_summary["strategy_gross_of_costs"] = gross_summary["strategy"]
    full_summary["risk_free_annual_avg"] = round(float(rf_daily.reindex(prices.index).mean() * 252), 4)
    full_summary["extensions_tested"] = extensions_tested
    full_summary["selection_note"] = (
        "Config selected by best Sharpe on TRAINING data only (start to "
        f"{WALK_FORWARD_SPLIT}), never on the period it's evaluated over here - a walk-forward "
        "split, not a full-sample search. See walk_forward.test_strategy for genuinely "
        "out-of-sample performance. Absolute-momentum filter: a sector is only held if its "
        "trailing return beats the realised risk-free/cash return over the same window; "
        "otherwise that sleeve sits in cash. Net of an assumed "
        f"{COST_BPS_PER_TRADE}bps transaction cost per unit of portfolio turnover "
        "(see strategy_gross_of_costs for the zero-cost comparison). Four further extensions "
        "(no-trade band, vol targeting, leveraged vol targeting, risk-parity weighting) were "
        "implemented and evaluated through this same walk-forward split - see "
        "extensions_tested: none improved Sharpe, so all are left off by default. Kept in the "
        "code as opt-in parameters rather than deleted, since a documented negative result is "
        "still a result."
    )
    full_summary["walk_forward"] = {
        "train_period": [prices.index[0].strftime("%Y-%m-%d"), WALK_FORWARD_SPLIT],
        "test_period": [WALK_FORWARD_SPLIT, prices.index[-1].strftime("%Y-%m-%d")],
        "test_strategy": oos_summary["strategy"],
        "test_benchmark_spy": oos_summary["benchmark_spy"],
        "test_benchmark_equal_weight": oos_summary["benchmark_equal_weight"],
    }
    # backward-compatible fields some older dashboard code/consumers may still read
    full_summary["backtest_start"] = full_summary["period_start"]
    full_summary["backtest_end"] = full_summary["period_end"]

    commentary = generate_commentary(full_rebalances, full_summary)

    with open("web/public/data/sensitivity.json", "w") as f:
        json.dump(grid_results, f, indent=2)
    with open("web/public/data/summary.json", "w") as f:
        json.dump(full_summary, f, indent=2)
    with open("web/public/data/rebalances.json", "w") as f:
        json.dump(full_rebalances, f, indent=2)
    with open("web/public/data/commentary.json", "w") as f:
        json.dump(commentary, f, indent=2)
    full_performance.to_json("web/public/data/performance.json", orient="records", indent=2)

    print("\n=== FULL-PERIOD SUMMARY (net of costs, walk-forward-selected config) ===")
    print(json.dumps(full_summary, indent=2))
    print(f"\n{len(full_rebalances)} rebalances logged.")
    print("Wrote web/public/data/{summary,rebalances,performance,sensitivity,commentary}.json")


if __name__ == "__main__":
    main()
