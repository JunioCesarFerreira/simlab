/**
 * The page's readings of a USL fit, and the planning that precedes one.
 *
 * The fit is computed and tested in `pylib/usl.py`. What is pinned here is
 * what the page *says* about it — when a peak is real, what a coefficient is
 * called, what a benchmark will cost — because that is what an operator acts
 * on.
 */
import { describe, expect, it } from "vitest";
import {
  benchmarkCost,
  chartData,
  formatCoefficient,
  interpret,
  parallelEfficiency,
  parseLevels,
  progress,
  proposeLevels,
  schedulablePool,
  usl,
  verdict,
  yAxisMax,
} from "../usl";
import type { UslFit, WaveMeasurement } from "../../types/scalability";

function fit(over: Partial<UslFit> = {}): UslFit {
  return {
    status: "ok",
    lam: 0.5,
    sigma: 0.05,
    kappa: 0,
    r2: 0.99,
    rmse: 0.01,
    n_points: 12,
    n_levels: 4,
    n_peak: null,
    n_peak_int: null,
    x_peak: null,
    x_asymptote: 10,
    peak_within_tested: false,
    levels: [],
    curve: [],
    ...over,
  };
}

function wave(level: number, batch: number | null, valid = true): WaveMeasurement {
  return {
    generation_id: `g${level}`,
    wave: 0,
    level,
    repetition: 0,
    status: "Done",
    simulations: level,
    done: valid ? level : 0,
    failed: valid ? 0 : level,
    hosts: { local: level },
    makespan_sec: 100,
    mean_held_sec: 100,
    throughput: { batch, per_simulation: batch },
    valid,
    reason: valid ? "" : "failed",
  };
}

describe("proposeLevels", () => {
  it("doubles up to the pool and ends at the whole pool", () => {
    expect(proposeLevels(14)).toEqual([1, 2, 4, 8, 14]);
  });

  it("does not repeat the pool when it is itself a power of two", () => {
    expect(proposeLevels(8)).toEqual([1, 2, 4, 8]);
  });

  it("has something to measure even with one container", () => {
    expect(proposeLevels(1)).toEqual([1]);
    expect(proposeLevels(0)).toEqual([1]);
  });
});

describe("parseLevels", () => {
  it("accepts commas, spaces and semicolons, and sorts", () => {
    expect(parseLevels("8, 1 4;2")).toEqual({ ok: true, levels: [1, 2, 4, 8] });
  });

  it("drops repeats", () => {
    expect(parseLevels("2,2,4")).toEqual({ ok: true, levels: [2, 4] });
  });

  it("refuses a fraction and says which token", () => {
    const result = parseLevels("1, 2.5");
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.error).toContain("2.5");
  });

  it("refuses zero, as the API does", () => {
    expect(parseLevels("0, 2").ok).toBe(false);
  });

  it("refuses nothing at all", () => {
    expect(parseLevels("  ").ok).toBe(false);
  });
});

describe("benchmarkCost", () => {
  it("counts waves and simulations before anything runs", () => {
    expect(benchmarkCost([1, 2, 4, 8], 3, 14)).toEqual({
      waves: 12,
      simulations: 45,
      beyondPool: [],
    });
  });

  it("names the levels larger than the pool", () => {
    // Their waves queue, so they measure the pool full rather than at N.
    expect(benchmarkCost([1, 8, 16], 1, 12).beyondPool).toEqual([16]);
  });
});

describe("formatCoefficient", () => {
  it("keeps an exact zero as zero — the fit found none", () => {
    expect(formatCoefficient(0)).toBe("0");
  });

  it("switches to exponent notation for the small values κ takes", () => {
    expect(formatCoefficient(0.00042)).toBe("4.20e-4");
    expect(formatCoefficient(0.0523)).toBe("0.0523");
  });
});

describe("verdict", () => {
  it("is retrograde only when the peak was actually reached", () => {
    expect(verdict(fit({ kappa: 0.004, n_peak: 6, peak_within_tested: true }))).toBe("retrograde");
  });

  it("is not retrograde for a peak past the tested range", () => {
    expect(verdict(fit({ kappa: 1e-4, n_peak: 90, peak_within_tested: false }))).toBe("contention");
  });

  it("calls negligible contention linear", () => {
    expect(verdict(fit({ sigma: 0.002, x_asymptote: 250 }))).toBe("linear");
  });

  it("does not guess without a fit", () => {
    expect(verdict(fit({ status: "insufficient_data" }))).toBe("unknown");
    expect(verdict(null)).toBe("unknown");
  });
});

describe("interpret", () => {
  it("states the peak, its throughput and that it was reached", () => {
    const lines = interpret(
      fit({
        kappa: 0.01,
        n_peak: 9.75,
        n_peak_int: 10,
        x_peak: 2.4,
        x_asymptote: null,
        peak_within_tested: true,
      }),
      14,
    );

    const peak = lines.find((l) => l.includes("N*"));
    expect(peak).toContain("9.8");
    expect(peak).toContain("best integer: 10");
    expect(peak).toContain("lowers throughput");
  });

  it("says when the peak is extrapolated", () => {
    const lines = interpret(
      fit({
        kappa: 1e-4,
        n_peak: 97,
        n_peak_int: 97,
        x_peak: 9,
        x_asymptote: null,
        peak_within_tested: false,
      }),
      14,
    );

    expect(lines.join(" ")).toContain("extrapolated");
  });

  it("states the ceiling when there is no peak", () => {
    expect(interpret(fit(), 8).join(" ")).toContain("approaches 10.0");
  });

  it("warns when only two levels could be fitted", () => {
    expect(interpret(fit({ status: "amdahl_only" }), 4).join(" ")).toContain("Add a third level");
  });

  it("flags a loose fit", () => {
    expect(interpret(fit({ r2: 0.5 }), 8).join(" ")).toContain("loose");
  });
});

describe("the model", () => {
  it("is λ at one unit of concurrency", () => {
    expect(usl(1, { lam: 0.7, sigma: 0.3, kappa: 0.01 })).toBeCloseTo(0.7);
  });

  it("gives full efficiency when nothing contends", () => {
    expect(parallelEfficiency(fit({ sigma: 0, kappa: 0, x_asymptote: null }), 12)).toBeCloseTo(1);
  });

  it("gives Amdahl's efficiency under pure contention", () => {
    // 1 / (1 + σ(N − 1)) at σ = 0.1, N = 11 → 0.5
    expect(parallelEfficiency(fit({ sigma: 0.1 }), 11)).toBeCloseTo(0.5);
  });
});

describe("chartData", () => {
  it("draws every valid repetition and skips failed waves", () => {
    const data = chartData(fit(), [wave(1, 0.5), wave(2, 0.9), wave(4, null, false)], "batch");

    expect(data.measured).toEqual([
      [1, 0.5],
      [2, 0.9],
    ]);
    expect(data.testedMax).toBe(4);
  });

  it("marks the peak with whether it was measured or extrapolated", () => {
    const data = chartData(
      fit({ n_peak: 6.2, n_peak_int: 6, x_peak: 2, peak_within_tested: true }),
      [wave(1, 0.5)],
      "batch",
    );

    expect(data.peak).toEqual({ n: 6, x: 2, withinTested: true });
  });

  it("has no curve without a fit, but still shows what was measured", () => {
    const data = chartData(fit({ status: "insufficient_data" }), [wave(2, 0.9)], "batch");

    expect(data.model).toEqual([]);
    expect(data.measured).toEqual([[2, 0.9]]);
  });

  it("keeps the lower error bar above zero", () => {
    const data = chartData(
      fit({ levels: [{ level: 1, mean: 0.1, std: 0.5, count: 2, min: 0, max: 0.6 }] }),
      [],
      "batch",
    );

    expect(data.means[0]?.low).toBe(0);
  });
});

describe("progress", () => {
  it("is the share of waves finished", () => {
    expect(progress({ waves: 12, waves_finished: 3 })).toBeCloseTo(0.25);
  });

  it("is zero for a benchmark with no plan", () => {
    expect(progress({ waves: 0, waves_finished: 0 })).toBe(0);
  });
});

describe("schedulablePool", () => {
  type Host = { schedulable: boolean; free: number; active: number };
  const overview = (hosts: Host[]) =>
    ({ hosts }) as unknown as Parameters<typeof schedulablePool>[0];

  it("counts only the hosts the dispatcher will place on", () => {
    // A disabled host keeps its containers `available`: the overview's totals
    // count them, and proposing them would plan waves that queue.
    expect(
      schedulablePool(
        overview([
          { schedulable: true, free: 6, active: 2 },
          { schedulable: false, free: 4, active: 0 },
        ]),
      ),
    ).toBe(8);
  });

  it("includes containers that are busy now", () => {
    expect(schedulablePool(overview([{ schedulable: true, free: 3, active: 5 }]))).toBe(8);
  });

  it("is zero with nothing to place on", () => {
    expect(schedulablePool(null)).toBe(0);
    expect(schedulablePool(overview([{ schedulable: false, free: 9, active: 0 }]))).toBe(0);
  });
});

describe("yAxisMax", () => {
  it("follows the data, not the linear reference", () => {
    // A degrading pool: data under 2, linear scaling reaching 14. Scaling to
    // the reference squashed the peak against the floor.
    const top = yAxisMax({
      measured: [
        [1, 0.5],
        [7, 1.63],
        [14, 1.4],
      ],
      means: [{ n: 7, mean: 1.63, low: 1.6, high: 1.66 }],
      model: [
        [1, 0.49],
        [7, 1.64],
        [28, 0.8],
      ],
    });

    expect(top).toBeGreaterThan(1.66);
    expect(top).toBeLessThan(3);
  });

  it("rounds to a readable step", () => {
    expect(yAxisMax({ measured: [[4, 14.7]], means: [], model: [] })).toBe(20);
    expect(yAxisMax({ measured: [[4, 1.63]], means: [], model: [] })).toBe(2.5);
  });

  it("has an axis even with nothing measured", () => {
    expect(yAxisMax({ measured: [], means: [], model: [] })).toBe(1);
  });
});
