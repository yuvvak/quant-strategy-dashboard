import fs from "fs";
import path from "path";

export type Stats = {
  total_return: number;
  annualized_return: number;
  annualized_vol: number;
  sharpe_ratio: number;
  max_drawdown: number;
};

export type Summary = {
  backtest_start: string;
  backtest_end: string;
  generated_at: string;
  config: { lookback_months: number; skip_month: boolean; top_n: number };
  strategy: Stats;
  benchmark_spy: Stats;
  benchmark_equal_weight: Stats;
  selection_note: string;
};

export type PerformancePoint = {
  date: string;
  strategy_cumulative: number;
  spy_cumulative: number;
  equal_weight_cumulative: number;
};

export type RebalanceEntry = {
  date: string;
  selected: { ticker: string; sector: string; momentum: number }[];
};

export type SensitivityRow = {
  lookback_months: number;
  skip_month: boolean;
  sharpe: number;
  annualized_return: number;
  vs_spy_sharpe: number;
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
