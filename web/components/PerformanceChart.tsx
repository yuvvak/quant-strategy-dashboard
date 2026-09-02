"use client";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { PerformancePoint } from "@/lib/data";

const COLORS = {
  strategy: "#2563eb",
  spy: "#6b7280",
  equalWeight: "#d97706",
};

function downsample<T>(arr: T[], maxPoints = 500): T[] {
  if (arr.length <= maxPoints) return arr;
  const step = Math.ceil(arr.length / maxPoints);
  return arr.filter((_, i) => i % step === 0);
}

export default function PerformanceChart({ data }: { data: PerformancePoint[] }) {
  const sampled = downsample(data);
  const chartData = sampled.map((d, i) => ({
    idx: i,
    date: d.date,
    Strategy: Number((d.strategy_cumulative * 100).toFixed(1)),
    SPY: Number((d.spy_cumulative * 100).toFixed(1)),
    "Equal-Weight Sectors": Number((d.equal_weight_cumulative * 100).toFixed(1)),
  }));

  const dateAt = (idx: number) => chartData[Math.round(idx)]?.date ?? "";

  return (
    <ResponsiveContainer width="100%" height={400}>
      <LineChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
        <XAxis
          dataKey="idx"
          type="number"
          domain={[0, chartData.length - 1]}
          tickFormatter={dateAt}
          tick={{ fontSize: 11, fill: "#6b7280" }}
          tickCount={7}
        />
        <YAxis
          tick={{ fontSize: 11, fill: "#6b7280" }}
          label={{ value: "Growth of 100", angle: -90, position: "insideLeft", fontSize: 12, fill: "#6b7280" }}
        />
        <Tooltip
          labelFormatter={(idx) => dateAt(Number(idx))}
          contentStyle={{ fontSize: 12, borderRadius: 8 }}
        />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Line
          type="monotone"
          dataKey="Strategy"
          stroke={COLORS.strategy}
          dot={false}
          strokeWidth={2}
          isAnimationActive={false}
        />
        <Line
          type="monotone"
          dataKey="SPY"
          stroke={COLORS.spy}
          dot={false}
          strokeWidth={1.5}
          isAnimationActive={false}
        />
        <Line
          type="monotone"
          dataKey="Equal-Weight Sectors"
          stroke={COLORS.equalWeight}
          dot={false}
          strokeWidth={1.5}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
