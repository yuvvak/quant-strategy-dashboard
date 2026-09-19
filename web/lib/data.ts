import fs from "fs";
import path from "path";

export type Stats = {
  total_return: number;
  annualized_return: number;
  annualized_vol: number;
  sharpe_ratio: number;
  max_drawdown: number;
};

export type WalkForward = {
  train_period: [string, string];
  test_period: [string, string];
  test_strategy: Stats;
  test_benchmark_spy: Stats;
  test_benchmark_equal_weight: Stats;
};

export type ExtensionTested = {
  label: string;
  full_sharpe: number;
  oos_sharpe: number;
  full_sharpe_delta: number;
  oos_sharpe_delta: number;
};

export type Summary = {
  backtest_start: string;
  backtest_end: string;
  period_start: string;
  period_end: string;
  generated_at: string;
  config: { lookback_months: number; skip_month: boolean; top_n: number };
  strategy: Stats;
  benchmark_spy: Stats;
  benchmark_equal_weight: Stats;
  strategy_gross_of_costs: Stats;
  transaction_cost_bps: number;
  risk_free_annual_avg: number;
  selection_note: string;
  walk_forward: WalkForward;
  extensions_tested: ExtensionTested[];
};

export type PerformancePoint = {
  date: string;
  strategy_cumulative: number;
  spy_cumulative: number;
  equal_weight_cumulative: number;
};

export type RebalanceEntry = {
  date: string;
  selected: { ticker: string; sector: string; momentum: number; weight: number }[];
  cash_weight: number;
  cash_hurdle: number;
  turnover: number;
  cost_charged: number;
};

export type SensitivityRow = {
  lookback_months: number;
  skip_month: boolean;
  sharpe: number;
  annualized_return: number;
  vs_spy_sharpe: number;
  selected: boolean;
};

export type Commentary = {
  generated_at: string;
  text: string;
  is_ai_generated: boolean;
  note: string;
};

function readJson<T>(filename: string): T {
  const filePath = path.join(process.cwd(), "public", "data", filename);
  return JSON.parse(fs.readFileSync(filePath, "utf-8")) as T;
}

export function getSummary(): Summary {
  return readJson<Summary>("summary.json");
}

export function getPerformance(): PerformancePoint[] {
  return readJson<PerformancePoint[]>("performance.json");
}

export function getRebalances(): RebalanceEntry[] {
  return readJson<RebalanceEntry[]>("rebalances.json");
}

export function getSensitivity(): SensitivityRow[] {
  return readJson<SensitivityRow[]>("sensitivity.json");
}

export function getCommentary(): Commentary {
  return readJson<Commentary>("commentary.json");
}
