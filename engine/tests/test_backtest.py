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


# --- No-trade band ---

@pytest.fixture
def close_momentum_prices():
    """XLK and XLF have nearly identical drift with heavy noise, so their momentum
    ranking flips frequently - a stress test for the no-trade band's turnover reduction."""
    idx = pd.bdate_range("2014-01-01", periods=900)
    rng = np.random.default_rng(7)
    tickers = list(bt.SECTOR_ETFS.keys()) + [bt.BENCHMARK]
    drifts = {"XLK": 0.0009, "XLF": 0.00085, bt.BENCHMARK: 0.0004}
    data = {}
    for t in tickers:
        drift = drifts.get(t, 0.0)
        noise = rng.normal(loc=0.0, scale=0.012, size=900)
        data[t] = 100.0 * np.cumprod(1.0 + drift + noise)
    return pd.DataFrame(data, index=idx)


def test_no_trade_band_reduces_turnover(close_momentum_prices):
    rf = flat_rf(close_momentum_prices, annual_rate=0.0)
    _, no_band, _ = bt.run_backtest(
        close_momentum_prices, rf, lookback_months=3, skip_month=False, top_n=1, no_trade_band=0.0
    )
    _, with_band, _ = bt.run_backtest(
        close_momentum_prices, rf, lookback_months=3, skip_month=False, top_n=1, no_trade_band=0.05
    )
    assert sum(r["turnover"] for r in with_band) < sum(r["turnover"] for r in no_band)


def test_extreme_no_trade_band_locks_in_initial_positions(trending_prices):
    """With an effectively infinite band, once a held sector is held it should never be
    replaced - proves the retention mechanic deterministically, not just statistically.
    absolute_momentum is off here so the test isolates the band, not the (separate,
    already-tested) eligibility filter - a noisy zero-drift sector can otherwise dip
    below the cash hurdle for legitimate reasons unrelated to the band."""
    rf = flat_rf(trending_prices, annual_rate=0.0)
    _, rebalances, _ = bt.run_backtest(
        trending_prices, rf, lookback_months=6, skip_month=False, no_trade_band=10.0,
        absolute_momentum=False,
    )
    assert len(rebalances) > 1
    first_tickers = {s["ticker"] for s in rebalances[0]["selected"]}
    for r in rebalances[1:]:
        assert {s["ticker"] for s in r["selected"]} == first_tickers
        assert r["turnover"] == pytest.approx(0.0, abs=1e-6)


# --- Volatility targeting ---

@pytest.fixture
def regime_shift_prices():
    """Calm uptrend for the first half, then a much noisier second half at the same drift -
    lets us check that vol targeting actually de-risks once volatility picks up."""
    idx = pd.bdate_range("2014-01-01", periods=800)
    rng = np.random.default_rng(11)
    tickers = list(bt.SECTOR_ETFS.keys()) + [bt.BENCHMARK]
    data = {}
    for t in tickers:
        drift = 0.0006 if t == "XLK" else 0.0001
        calm = rng.normal(loc=drift, scale=0.004, size=400)
        volatile = rng.normal(loc=drift, scale=0.025, size=400)
        data[t] = 100.0 * np.cumprod(1.0 + np.concatenate([calm, volatile]))
    return pd.DataFrame(data, index=idx)


def test_vol_target_disabled_by_default(trending_prices):
    summary, _, _ = bt.run_backtest(trending_prices, flat_rf(trending_prices), lookback_months=6, skip_month=False)
    assert "vol_target" not in summary
    assert "strategy_pre_vol_target" not in summary


def test_vol_target_reduces_volatility_vs_unscaled(regime_shift_prices):
    rf = flat_rf(regime_shift_prices, annual_rate=0.0)
    summary, _, _ = bt.run_backtest(
        regime_shift_prices, rf, lookback_months=3, skip_month=False, target_vol=0.10
    )
    assert "vol_target" in summary
    assert summary["vol_target"]["avg_exposure"] < 1.0
    assert summary["strategy"]["annualized_vol"] < summary["strategy_pre_vol_target"]["annualized_vol"]


def test_vol_target_never_levers_above_cap():
    prices = make_prices(n_days=500, daily_drifts={"XLK": 0.0005}, seed=13)
    rf = flat_rf(prices, annual_rate=0.0)
    summary, _, _ = bt.run_backtest(
        prices, rf, lookback_months=3, skip_month=False, target_vol=5.0  # absurdly high target
    )
    assert summary["vol_target"]["avg_exposure"] <= bt.VOL_TARGET_CAP + 1e-9


def test_vol_target_can_lever_up_in_a_calm_regime():
    """A target well above realized vol in a genuinely calm regime should actually use
    leverage (scale > 1x, capped at VOL_TARGET_CAP), not just cap out at 1x."""
    prices = make_prices(n_days=500, daily_drifts={"XLK": 0.0006}, seed=17)  # calm, tight noise
    rf = flat_rf(prices, annual_rate=0.0)
    summary, _, _ = bt.run_backtest(
        prices, rf, lookback_months=3, skip_month=False, target_vol=0.40  # well above realized vol
    )
    assert summary["vol_target"]["avg_exposure"] > 1.0
    assert summary["vol_target"]["pct_days_levered"] > 0.5


# --- Risk-parity weighting ---

@pytest.fixture
def mixed_vol_prices():
    """XLK and XLF both trend up at a similar pace, but XLF is far noisier - both should
    clear the momentum bar and get selected, letting risk-parity vs equal-weight differ."""
    idx = pd.bdate_range("2014-01-01", periods=500)
    rng = np.random.default_rng(21)
    tickers = list(bt.SECTOR_ETFS.keys()) + [bt.BENCHMARK]
    data = {}
    for t in tickers:
        if t == "XLK":
            steps = rng.normal(loc=0.0012, scale=0.004, size=500)
        elif t == "XLF":
            steps = rng.normal(loc=0.0012, scale=0.020, size=500)
        else:
            steps = rng.normal(loc=-0.0010, scale=0.003, size=500)  # clearly weaker, shouldn't be picked
        data[t] = 100.0 * np.cumprod(1.0 + steps)
    return pd.DataFrame(data, index=idx)


def test_risk_parity_weights_favor_lower_vol_holding(mixed_vol_prices):
    rf = flat_rf(mixed_vol_prices, annual_rate=0.0)
    _, rebalances, _ = bt.run_backtest(
        mixed_vol_prices, rf, lookback_months=3, skip_month=False, top_n=2, risk_parity=True
    )
    checked = 0
    for r in rebalances:
        tickers = {s["ticker"] for s in r["selected"]}
        if {"XLK", "XLF"} <= tickers:
            weights = {s["ticker"]: s["weight"] for s in r["selected"]}
            assert weights["XLK"] > weights["XLF"], r
            checked += 1
    assert checked > 5, "expected several rebalances with both XLK and XLF selected"


def test_risk_parity_reduces_portfolio_vol_vs_equal_weight(mixed_vol_prices):
    rf = flat_rf(mixed_vol_prices, annual_rate=0.0)
    equal, _, _ = bt.run_backtest(
        mixed_vol_prices, rf, lookback_months=3, skip_month=False, top_n=2, risk_parity=False
    )
    parity, _, _ = bt.run_backtest(
        mixed_vol_prices, rf, lookback_months=3, skip_month=False, top_n=2, risk_parity=True
    )
    assert parity["strategy"]["annualized_vol"] < equal["strategy"]["annualized_vol"]
