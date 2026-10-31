/**
 * What the page says about a seed study, and the planning before one.
 *
 * The statistics are computed and tested in `pylib/seed_study.py`.
 */
import { describe, expect, it } from "vitest";
import {
  absoluteParam,
  checkSeeds,
  estimateHours,
  formatHours,
  formatInterval,
  formatValue,
  headline,
  metricStatus,
  nearZeroMetrics,
  normalityLabel,
  progress,
  scopeSentence,
  statusInfo,
  studyCost,
} from "../seedStudy";
import {
  bySolutionOption,
  histogramOption,
  precisionOption,
  reliabilityOption,
} from "../seedStudyCharts";
import { chartPalette } from "../chartTheme";
import type { MetricDetail, StudyDetail, StudyMetric } from "../../types/seedStudy";

function detail(over: Partial<StudyDetail> = {}): StudyDetail {
  return {
    problem: "problem1",
    duration: 180,
    provenance: { images: ["cooja:v1.1"], firmware: null, seeds: null, panel: {} },
    analysis: {
      shape: { chromosomes: 20, seeds: 30 },
      recommendation: { optimise: { max: 8, by_objective: {} } },
    },
    ...over,
  } as unknown as StudyDetail;
}

describe("studyCost", () => {
  it("counts the control apart", () => {
    expect(studyCost(30, 20, 2)).toEqual({ main: 600, control: 40, total: 640, overLimit: false });
  });

  it("applies the limit the server applies, on the main runs", () => {
    expect(studyCost(60, 100, 2).overLimit).toBe(true);
    expect(studyCost(50, 100, 2).overLimit).toBe(false);
  });
});

describe("estimateHours", () => {
  it("divides the work over the pool", () => {
    expect(estimateHours(640, 145, 40)).toBeCloseTo(0.644, 2);
  });

  it("never promises less than one simulation takes", () => {
    expect(estimateHours(5, 360, 12)).toBeCloseTo(0.1, 3);
  });

  it("says nothing with no history or no pool", () => {
    expect(estimateHours(640, null, 40)).toBeNull();
    expect(estimateHours(640, 145, 0)).toBeNull();
  });

  it("reads in the unit that suits it", () => {
    expect(formatHours(0.12)).toBe("7 min");
    expect(formatHours(2.6)).toBe("2.6 h");
  });
});

describe("checkSeeds", () => {
  it("accepts a clean list", () => {
    expect(checkSeeds([1000, 20000, 40000])).toEqual({
      errors: [],
      lowCollisions: 0,
      closePairs: 0,
    });
  });

  it("refuses what the server refuses", () => {
    expect(checkSeeds([1, 2]).errors).toHaveLength(1);
    expect(checkSeeds([1000, 1000, 40000]).errors).toEqual(["The list repeats a seed."]);
    expect(checkSeeds([0, 20000, 40000]).errors).toHaveLength(1);
  });

  it("only warns about structure: the list may be a reproduction", () => {
    const check = checkSeeds([1, 2, 3]);

    expect(check.errors).toEqual([]);
    expect(check.closePairs).toBe(3);
  });

  it("sees seeds that differ only above the low 16 bits", () => {
    expect(checkSeeds([5, 5 + 65536, 30000]).lowCollisions).toBe(1);
  });
});

describe("progress", () => {
  it("is a share of the simulations, never past one", () => {
    expect(progress({ finished: 32, simulations: 640 })).toBeCloseTo(0.05);
    expect(progress({ finished: 700, simulations: 640 })).toBe(1);
    expect(progress({ simulations: 0 })).toBe(0);
  });
});

describe("formatting", () => {
  it.each([
    [null, "—"],
    [0, "0"],
    [213.4567, "213.5"],
    [5456000, "5,456,000"],
    [1.82e19, "1.82e+19"],
    [0.0001, "1.00e-4"],
  ])("formats %s", (value, text) => {
    expect(formatValue(value)).toBe(text);
  });

  it("formats an interval", () => {
    expect(formatInterval([183.5, 243.3])).toBe("[183.5, 243.3]");
    expect(formatInterval(null)).toBe("—");
  });
});

describe("absolute targets", () => {
  it("are sent as the route reads them", () => {
    expect(absoluteParam({ latency: 5, delta: 0.5 })).toBe("latency:5,delta:0.5");
  });

  it("are left out when there are none", () => {
    expect(absoluteParam({})).toBeUndefined();
    expect(absoluteParam()).toBeUndefined();
  });

  it("are asked for where a relative error means nothing", () => {
    const metric = (name: string, statuses: string[]) =>
      ({ name, summary: { statuses } }) as unknown as StudyMetric;

    expect(nearZeroMetrics([metric("a", ["ok"]), metric("b", ["near_zero_mean"])])).toEqual(["b"]);
  });
});

describe("normalityLabel", () => {
  it("reads Shapiro–Wilk without promising normality", () => {
    expect(normalityLabel(null)).toBe("—");
    expect(normalityLabel(0)).toBe("not rejected");
    expect(normalityLabel(0.2)).toBe("doubtful");
    expect(normalityLabel(0.7)).toBe("rejected");
  });
});

describe("statuses", () => {
  it("names a defect as a defect", () => {
    expect(statusInfo("overflow_suspect").tone).toBe("bad");
  });

  it("falls back on the code for an unknown status", () => {
    expect(statusInfo("something_new").label).toBe("something_new");
  });

  it("shows the metric's own status before anything about n*", () => {
    const metric = { status: "constant", summary: { statuses: [] } } as unknown as StudyMetric;

    expect(metricStatus(metric)).toBe("constant");
  });

  it("uses the recommendation of an objective", () => {
    const metric = {
      status: "ok",
      summary: { statuses: ["indicative"] },
    } as unknown as StudyMetric;

    expect(metricStatus(metric, { status: "ok", value: 8, p90: 9, max: 9, method: "t" })).toBe(
      "ok",
    );
  });
});

describe("the sentences of the recommendation", () => {
  it("always says what the number is valid for", () => {
    expect(scopeSentence(detail())).toBe(
      "Valid for problem1, 180 min simulated, image cooja:v1.1.",
    );
  });

  it("gives the number of seeds to optimise", () => {
    expect(headline(detail())).toBe("Use 8 seeds to optimise.");
  });

  it("says one seed in the singular", () => {
    const flat = detail({
      analysis: {
        shape: { chromosomes: 20, seeds: 30 },
        recommendation: { optimise: { max: 1, by_objective: {} } },
      },
    } as unknown as Partial<StudyDetail>);

    expect(headline(flat)).toBe("Use 1 seed to optimise.");
  });

  it("does not answer ranking from one solution", () => {
    const one = detail({
      analysis: {
        shape: { chromosomes: 1, seeds: 30 },
        recommendation: { optimise: { max: null, by_objective: {} } },
      },
    } as unknown as Partial<StudyDetail>);

    expect(headline(one)).toContain("Ranking solutions needs a panel");
  });

  it("says when a panel gives no answer", () => {
    const flat = detail({
      analysis: {
        shape: { chromosomes: 20, seeds: 30 },
        recommendation: { optimise: { max: null, by_objective: {} } },
      },
    } as unknown as Partial<StudyDetail>);

    expect(headline(flat)).toContain("no number of seeds");
  });

  it("has nothing to say before a result", () => {
    expect(headline(detail({ analysis: null }))).toBe("No result yet.");
  });
});

describe("chart options", () => {
  const c = chartPalette(false);

  it("draws one bar per bin", () => {
    const option = histogramOption(
      {
        metric: "latency",
        histogram: [
          { from: 0, to: 1, count: 2 },
          { from: 1, to: 2, count: 3 },
        ],
      } as MetricDetail,
      c,
    );

    expect((option.series as { data: number[] }[])[0]!.data).toEqual([2, 3]);
  });

  it("draws P(k) against the confidence asked for", () => {
    const option = precisionOption(
      { metric: "latency", precision: { curve: [{ k: 2, p: 0.4, expected: 9 }] } } as MetricDetail,
      0.95,
      false,
      c,
    );

    const series = (option.series as { data: number[][]; markLine: { data: unknown[] } }[])[0]!;
    expect(series.data).toEqual([[2, 0.4]]);
    expect(series.markLine.data).toEqual([{ yAxis: 0.95 }]);
  });

  it("draws the moving target of an extreme instead of a probability", () => {
    const option = precisionOption(
      { metric: "latency", precision: { curve: [{ k: 2, p: 0.4, expected: 9 }] } } as MetricDetail,
      0.95,
      true,
      c,
    );

    expect((option.series as { data: number[][] }[])[0]!.data).toEqual([[2, 9]]);
  });

  it("draws both reliabilities and the ranking that checks them", () => {
    const option = reliabilityOption(
      {
        reliability: [{ m: 1, e_rho2: 0.6, phi: 0.5 }],
        rank_stability: [{ k: 1, spearman: 0.55 }],
      } as never,
      0.9,
      c,
    );

    expect((option.series as unknown[]).length).toBe(3);
  });

  it("sorts solutions by median and leaves out the ones with no data", () => {
    const row = (median: number, n = 3) => ({
      genome_hash: `h${median}`,
      n,
      p5: median - 2,
      q1: median - 1,
      median,
      q3: median + 1,
      p95: median + 2,
    });
    const option = bySolutionOption(
      { name: "latency", by_chromosome: [row(9), row(1), row(5, 0)] } as unknown as StudyMetric,
      c,
    );

    const series = (option.series as { data: number[][] }[])[0]!;
    expect(series.data.map((box) => box[2])).toEqual([1, 9]);
  });
});
