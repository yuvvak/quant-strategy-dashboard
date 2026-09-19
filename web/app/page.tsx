import PerformanceChart from "@/components/PerformanceChart";
import {
  getCommentary,
  getPerformance,
  getRebalances,
  getSensitivity,
  getSummary,
  type Stats,
} from "@/lib/data";

function pct(v: number, digits = 1) {
  return `${(v * 100).toFixed(digits)}%`;
}

function StatCard({ label, value, positive }: { label: string; value: string; positive?: boolean }) {
  return (
    <div className="rounded-xl border border-gray-200 bg-white p-4">
      <div className="text-xs uppercase tracking-wide text-gray-500">{label}</div>
      <div
        className={`mt-1 text-xl font-semibold ${
          positive === undefined ? "text-gray-900" : positive ? "text-emerald-600" : "text-red-600"
        }`}
      >
        {value}
      </div>
    </div>
  );
}

function StrategyRow({ name, stats, highlight }: { name: string; stats: Stats; highlight?: boolean }) {
  return (
    <tr className={highlight ? "bg-blue-50 font-medium" : ""}>
      <td className="px-3 py-2 text-sm">{name}</td>
      <td className="px-3 py-2 text-sm text-right">{pct(stats.total_return)}</td>
      <td className="px-3 py-2 text-sm text-right">{pct(stats.annualized_return)}</td>
      <td className="px-3 py-2 text-sm text-right">{pct(stats.annualized_vol)}</td>
      <td className="px-3 py-2 text-sm text-right">{stats.sharpe_ratio.toFixed(2)}</td>
      <td className="px-3 py-2 text-sm text-right">{pct(stats.max_drawdown)}</td>
    </tr>
  );
}

export default function Home() {
  const summary = getSummary();
  const performance = getPerformance();
  const rebalances = getRebalances();
  const sensitivity = getSensitivity();
  const commentary = getCommentary();
  const recentRebalances = [...rebalances].reverse().slice(0, 6);

  return (
    <main className="mx-auto max-w-5xl px-4 py-10 sm:px-6">
      <header className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900 sm:text-3xl">
          Sector Momentum Rotation Strategy
        </h1>
        <p className="mt-1 text-sm text-gray-600">
          Top {summary.config.top_n} of 11 SPDR sector ETFs by {summary.config.lookback_months}
          -month price momentum{summary.config.skip_month ? " (skipping the most recent month)" : ""},
          equal-weighted, rebalanced monthly, with an absolute-momentum/cash overlay. Config
          selected via a walk-forward split - see below. Backtested {summary.backtest_start} to{" "}
          {summary.backtest_end}, net of a {summary.transaction_cost_bps}bps transaction-cost
          assumption per unit of turnover.
        </p>
      </header>

      <section className="mb-8 rounded-xl border border-blue-100 bg-blue-50/60 p-4 text-sm text-gray-800">
        <div className="mb-1 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-blue-700">
          Latest state
          <span className="rounded-full bg-white px-2 py-0.5 text-[10px] font-medium text-gray-500">
            rule-based, not AI-generated
          </span>
        </div>
        <p>{commentary.text}</p>
      </section>

      <section className="mb-8 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <StatCard label="Total Return (net)" value={pct(summary.strategy.total_return)} />
        <StatCard label="Annualized Return (net)" value={pct(summary.strategy.annualized_return)} />
        <StatCard label="Sharpe Ratio (net)" value={summary.strategy.sharpe_ratio.toFixed(2)} />
        <StatCard label="Max Drawdown" value={pct(summary.strategy.max_drawdown)} positive={false} />
      </section>

      <section className="mb-8 rounded-xl border border-gray-200 bg-white p-4 text-sm text-gray-700">
        <h2 className="mb-2 text-sm font-semibold text-gray-700">Gross vs. net of transaction costs</h2>
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <div>
            <div className="text-xs uppercase tracking-wide text-gray-500">Gross Ann. Return</div>
            <div className="mt-0.5 font-medium">{pct(summary.strategy_gross_of_costs.annualized_return)}</div>
          </div>
          <div>
            <div className="text-xs uppercase tracking-wide text-gray-500">Net Ann. Return</div>
            <div className="mt-0.5 font-medium">{pct(summary.strategy.annualized_return)}</div>
          </div>
          <div>
            <div className="text-xs uppercase tracking-wide text-gray-500">Gross Sharpe</div>
            <div className="mt-0.5 font-medium">{summary.strategy_gross_of_costs.sharpe_ratio.toFixed(2)}</div>
          </div>
          <div>
            <div className="text-xs uppercase tracking-wide text-gray-500">Net Sharpe</div>
            <div className="mt-0.5 font-medium">{summary.strategy.sharpe_ratio.toFixed(2)}</div>
          </div>
        </div>
        <p className="mt-2 text-xs text-gray-500">
          Cost assumption: {summary.transaction_cost_bps}bps per unit of portfolio turnover per
          rebalance (turnover includes moving to/from cash). Risk-free rate used in every Sharpe
          ratio here (and as the absolute-momentum cash hurdle): real 13-week T-bill yield,
          averaging {pct(summary.risk_free_annual_avg)} annualized over the backtest.
        </p>
      </section>

      <section className="mb-8 rounded-xl border border-gray-200 bg-white p-4">
        <h2 className="mb-1 text-sm font-semibold text-gray-700">
          Walk-forward out-of-sample test
        </h2>
        <p className="mb-3 text-xs text-gray-500">
          Config chosen using only {summary.walk_forward.train_period[0]} to{" "}
          {summary.walk_forward.train_period[1]} (training). Stats below are the same config
          evaluated only on {summary.walk_forward.test_period[0]} to{" "}
          {summary.walk_forward.test_period[1]} - a period never used to pick the config, i.e.
          genuinely out-of-sample, not the full-history numbers above.
        </p>
        <table className="w-full">
          <thead>
            <tr className="border-b border-gray-200 text-left text-xs uppercase tracking-wide text-gray-500">
              <th className="px-3 py-2">Portfolio</th>
              <th className="px-3 py-2 text-right">Total Return</th>
              <th className="px-3 py-2 text-right">Ann. Return</th>
              <th className="px-3 py-2 text-right">Sharpe</th>
              <th className="px-3 py-2 text-right">Max Drawdown</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            <tr className="bg-blue-50 font-medium">
              <td className="px-3 py-2 text-sm">Strategy (out-of-sample)</td>
              <td className="px-3 py-2 text-right text-sm">{pct(summary.walk_forward.test_strategy.total_return)}</td>
              <td className="px-3 py-2 text-right text-sm">{pct(summary.walk_forward.test_strategy.annualized_return)}</td>
              <td className="px-3 py-2 text-right text-sm">{summary.walk_forward.test_strategy.sharpe_ratio.toFixed(2)}</td>
              <td className="px-3 py-2 text-right text-sm">{pct(summary.walk_forward.test_strategy.max_drawdown)}</td>
            </tr>
            <tr>
              <td className="px-3 py-2 text-sm">SPY (out-of-sample)</td>
              <td className="px-3 py-2 text-right text-sm">{pct(summary.walk_forward.test_benchmark_spy.total_return)}</td>
              <td className="px-3 py-2 text-right text-sm">{pct(summary.walk_forward.test_benchmark_spy.annualized_return)}</td>
              <td className="px-3 py-2 text-right text-sm">{summary.walk_forward.test_benchmark_spy.sharpe_ratio.toFixed(2)}</td>
              <td className="px-3 py-2 text-right text-sm">{pct(summary.walk_forward.test_benchmark_spy.max_drawdown)}</td>
            </tr>
          </tbody>
        </table>
      </section>

      <section className="mb-8 rounded-xl border border-gray-200 bg-white p-4">
        <h2 className="mb-3 text-sm font-semibold text-gray-700">
          Growth of 100, Strategy vs. Benchmarks
        </h2>
        <PerformanceChart data={performance} />
      </section>

      <section className="mb-8 overflow-x-auto rounded-xl border border-gray-200 bg-white">
        <table className="w-full">
          <thead>
            <tr className="border-b border-gray-200 text-left text-xs uppercase tracking-wide text-gray-500">
              <th className="px-3 py-2">Portfolio</th>
              <th className="px-3 py-2 text-right">Total Return</th>
              <th className="px-3 py-2 text-right">Ann. Return</th>
              <th className="px-3 py-2 text-right">Volatility</th>
              <th className="px-3 py-2 text-right">Sharpe</th>
              <th className="px-3 py-2 text-right">Max Drawdown</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            <StrategyRow name="Strategy" stats={summary.strategy} highlight />
            <StrategyRow name="SPY (buy &amp; hold)" stats={summary.benchmark_spy} />
            <StrategyRow name="Equal-Weight Sectors" stats={summary.benchmark_equal_weight} />
          </tbody>
        </table>
      </section>

      <section className="mb-8 grid gap-6 sm:grid-cols-2">
        <div className="rounded-xl border border-gray-200 bg-white p-4">
          <h2 className="mb-3 text-sm font-semibold text-gray-700">Most Recent Rebalances</h2>
          <ul className="space-y-3">
            {recentRebalances.map((r) => (
              <li key={r.date} className="text-sm">
                <div className="font-medium text-gray-800">{r.date}</div>
                <div className="mt-0.5 flex flex-wrap gap-1.5">
                  {r.selected.map((s) => (
                    <span
                      key={s.ticker}
                      className="rounded-full bg-blue-50 px-2 py-0.5 text-xs text-blue-700"
                      title={`${s.sector}: ${pct(s.momentum)}`}
                    >
                      {s.ticker} {pct(s.momentum)}
                    </span>
                  ))}
                  {r.cash_weight > 0.001 && (
                    <span
                      className="rounded-full bg-amber-50 px-2 py-0.5 text-xs text-amber-700"
                      title={`Cash hurdle this period: ${pct(r.cash_hurdle)}`}
                    >
                      Cash {pct(r.cash_weight, 0)}
                    </span>
                  )}
                </div>
              </li>
            ))}
          </ul>
        </div>

        <div className="rounded-xl border border-gray-200 bg-white p-4">
          <h2 className="mb-3 text-sm font-semibold text-gray-700">
            Sensitivity Grid (lookback &times; skip-month)
          </h2>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs uppercase tracking-wide text-gray-500">
                <th className="py-1">Lookback</th>
                <th className="py-1">Skip 1mo</th>
                <th className="py-1 text-right">Sharpe</th>
                <th className="py-1 text-right">vs SPY</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {sensitivity.map((row) => (
                <tr key={`${row.lookback_months}-${row.skip_month}`} className={row.selected ? "font-medium" : ""}>
                  <td className="py-1">{row.lookback_months}mo</td>
                  <td className="py-1">{row.skip_month ? "Yes" : "No"}</td>
                  <td className="py-1 text-right">{row.sharpe.toFixed(2)}</td>
                  <td className={`py-1 text-right ${row.vs_spy_sharpe >= 0 ? "text-emerald-600" : "text-gray-500"}`}>
                    {row.vs_spy_sharpe >= 0 ? "+" : ""}
                    {row.vs_spy_sharpe.toFixed(3)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-2 text-xs text-gray-500">
            Grid evaluated on the training period only ({summary.walk_forward.train_period[0]} to{" "}
            {summary.walk_forward.train_period[1]}) - see the walk-forward section above for how
            the selected config performed afterward.
          </p>
        </div>
      </section>

      <footer className="rounded-xl border border-dashed border-gray-300 bg-gray-50 p-4 text-xs text-gray-600">
        <p className="mb-1 font-medium text-gray-700">Methodology &amp; limitations</p>
        <p>{summary.selection_note}</p>
        <p className="mt-1">
          Backtest models transaction costs (turnover-based) and a real risk-free rate, but still
          does not model slippage beyond the flat cost assumption, taxes, or dividend timing
          differences across the {rebalances.length} rebalances. Live implementation would still
          likely underperform this backtest by some margin. Sector ETF universe: 11 SPDR Select
          Sector SPDRs. Data recomputed on a schedule via GitHub Actions - see the repo README for
          the run history. Generated {new Date(summary.generated_at).toLocaleString()}.
        </p>
      </footer>
    </main>
  );
}
