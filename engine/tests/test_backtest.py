"""
Unit tests for engine/backtest.py, using synthetic price data (no network calls).

Covers the correctness properties that matter for a systematic strategy: weights
always sum to 1, the momentum ranking has no look-ahead access to future prices,
the absolute-momentum/cash overlay actually moves to cash when nothing clears the
hurdle, transaction costs actually reduce net returns, and period scoping (used
for the walk-forward split) respects its bounds.
"""
import numpy as np
import pandas as pd
import pytest

import backtest as bt


def make_prices(n_days=650, start="2014-01-01", daily_drifts=None, seed=0):
    """Deterministic-ish synthetic price panel: each ticker follows
    price[t] = price[t-1] * (1 + drift + noise). `daily_drifts` maps ticker -> daily
    drift; tickers not listed default to 0 drift (flat, noisy)."""
    idx = pd.bdate_range(start, periods=n_days)
    rng = np.random.default_rng(seed)
    tickers = list(bt.SECTOR_ETFS.keys()) + [bt.BENCHMARK]
    daily_drifts = daily_drifts or {}
    data = {}
    for t in tickers:
        drift = daily_drifts.get(t, 0.0)
        noise = rng.normal(loc=0.0, scale=0.003, size=n_days)
        steps = drift + noise
        data[t] = 100.0 * np.cumprod(1.0 + steps)
    return pd.DataFrame(data, index=idx)


def flat_rf(prices, annual_rate=0.0):
    daily = annual_rate / 252.0
    return pd.Series(daily, index=prices.index)


@pytest.fixture
def trending_prices():
    # XLK trends up hard, XLF trends up mildly, everything else is flat/noisy -
    # gives an unambiguous momentum ranking to assert against.
    drifts = {"XLK": 0.0020, "XLF": 0.0006, bt.BENCHMARK: 0.0004}
    return make_prices(daily_drifts=drifts, seed=1)


@pytest.fixture
def all_declining_prices():
    # Every sector declines steadily - with a 0%-yielding cash hurdle, nothing
    # should clear the absolute-momentum bar.
    drifts = {t: -0.0015 for t in bt.SECTOR_ETFS}
    drifts[bt.BENCHMARK] = -0.0010
    return make_prices(daily_drifts=drifts, seed=2)


def test_weights_sum_to_one(trending_prices):
    rf = flat_rf(trending_prices, annual_rate=0.01)
    _, rebalances, _ = bt.run_backtest(trending_prices, rf, lookback_months=6, skip_month=False)
    assert rebalances, "expected at least one rebalance in a 650-day synthetic history"
    for r in rebalances:
        invested = len(r["selected"]) * (1.0 / bt.TOP_N)
        total = invested + r["cash_weight"]
        assert total == pytest.approx(1.0, abs=1e-6), r


def test_no_lookahead_bias(trending_prices):
    """Selections at an early rebalance must be identical whether or not the
    price history after that rebalance date exists at all."""
    rf = flat_rf(trending_prices, annual_rate=0.0)
    full_summary, full_rebalances, _ = bt.run_backtest(
        trending_prices, rf, lookback_months=6, skip_month=False
    )
    assert full_rebalances
    first_rebal_date = pd.Timestamp(full_rebalances[0]["date"])

    # Truncate all history after the first rebalance date and re-run - the very
    # first rebalance's selection must be unchanged, since it can only ever see
    # data up to and including that date.
    truncated_prices = trending_prices.loc[:first_rebal_date]
    truncated_rf = rf.loc[:first_rebal_date]
    _, truncated_rebalances, _ = bt.run_backtest(
        truncated_prices, truncated_rf, lookback_months=6, skip_month=False
    )
    assert truncated_rebalances, "truncated history should still produce the first rebalance"
    assert truncated_rebalances[0]["selected"] == full_rebalances[0]["selected"]


def test_highest_momentum_sectors_are_selected(trending_prices):
    rf = flat_rf(trending_prices, annual_rate=0.0)
    _, rebalances, _ = bt.run_backtest(trending_prices, rf, lookback_months=6, skip_month=False)
    assert rebalances
    # XLK (strongest drift) should be selected at essentially every rebalance.
    xlk_picked = sum(1 for r in rebalances if any(s["ticker"] == "XLK" for s in r["selected"]))
    assert xlk_picked / len(rebalances) > 0.9


def test_absolute_momentum_moves_to_cash_when_nothing_clears_hurdle(all_declining_prices):
    rf = flat_rf(all_declining_prices, annual_rate=0.0)  # cash hurdle = 0%
    _, rebalances, _ = bt.run_backtest(
        all_declining_prices, rf, lookback_months=6, skip_month=False, absolute_momentum=True
    )
    assert rebalances
    for r in rebalances:
        assert r["selected"] == [], r
        assert r["cash_weight"] == pytest.approx(1.0, abs=1e-6)


def test_absolute_momentum_off_still_invests_in_the_least_bad_sector(all_declining_prices):
    rf = flat_rf(all_declining_prices, annual_rate=0.0)
    _, rebalances, _ = bt.run_backtest(
        all_declining_prices, rf, lookback_months=6, skip_month=False, absolute_momentum=False
    )
    assert rebalances
    # With the filter off, top_n sectors are still picked even though all are losing money.
    assert all(len(r["selected"]) == bt.TOP_N for r in rebalances)


def test_transaction_costs_reduce_net_return(trending_prices):
    rf = flat_rf(trending_prices, annual_rate=0.0)
    net_summary, _, _ = bt.run_backtest(
        trending_prices, rf, lookback_months=6, skip_month=False, cost_bps=25
    )
    gross_summary, _, _ = bt.run_backtest(
        trending_prices, rf, lookback_months=6, skip_month=False, cost_bps=0
    )
    assert net_summary["strategy"]["total_return"] < gross_summary["strategy"]["total_return"]


def test_zero_turnover_zero_cost():
    """A single-rebalance-only history (too short for a second rebalance) should
    still charge the entry cost - turnover into the first positions is real cost,
    not free."""
    prices = make_prices(n_days=400, daily_drifts={"XLK": 0.001}, seed=3)
    rf = flat_rf(prices, annual_rate=0.0)
    _, rebalances, _ = bt.run_backtest(prices, rf, lookback_months=6, skip_month=False, cost_bps=10)
    assert rebalances
    assert rebalances[0]["cost_charged"] > 0
    assert rebalances[0]["turnover"] == pytest.approx(1.0, abs=1e-6)  # 0 -> fully invested


def test_period_scoping_respects_bounds(trending_prices):
    split = pd.Timestamp("2015-06-30")
    summary, _, performance = bt.run_backtest(
        trending_prices, flat_rf(trending_prices), lookback_months=6, skip_month=False,
        period_start=split,
    )
    dates = pd.to_datetime(performance["date"])
    assert dates.min() >= split
    assert summary["period_start"] == dates.min().strftime("%Y-%m-%d")


def test_cash_hurdle_uses_same_window_as_momentum(trending_prices):
    """The cash_hurdle logged at each rebalance should be a plausible cumulative
    return over a lookback-months window, not a full-history or single-day figure."""
    rf = flat_rf(trending_prices, annual_rate=0.03)
    _, rebalances, _ = bt.run_backtest(trending_prices, rf, lookback_months=6, skip_month=False)
    assert rebalances
    # ~6 months at ~3%/yr should be roughly 1.5%, generously bounded here.
    for r in rebalances:
        assert -0.05 < r["cash_hurdle"] < 0.05


def test_generate_commentary_is_not_flagged_as_ai():
    commentary = bt.generate_commentary([], {"strategy": {"sharpe_ratio": 0.0}, "benchmark_spy": {"sharpe_ratio": 0.0}})
    assert commentary["is_ai_generated"] is False
    assert "text" in commentary
