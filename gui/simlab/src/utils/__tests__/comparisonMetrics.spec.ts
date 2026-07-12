import { describe, it, expect } from "vitest";
import {
  extractFront,
  dominates,
  coverage,
  epsilonIndicator,
  igdPlus,
  spacing,
} from "../comparisonMetrics";
import { PENALTY_THRESHOLD } from "../../types/simlab";
import type { ObjectiveItem, ParetoFrontItemDto } from "../../types/simlab";

const MIN2 = [true, true];

describe("extractFront", () => {
  const objectives: ObjectiveItem[] = [
    { metric_name: "latency", goal: "min" },
    { metric_name: "energy", goal: "min" },
  ];

  it("projects items onto the objective order and drops penalized points", () => {
    const items: ParetoFrontItemDto[] = [
      { chromosome: {}, objectives: { energy: 2, latency: 1 } },
      { chromosome: {}, objectives: { latency: PENALTY_THRESHOLD, energy: 0 } },
    ];
    expect(extractFront(items, objectives)).toEqual([[1, 2]]);
  });

  it("fills missing metrics with 0", () => {
    const items: ParetoFrontItemDto[] = [{ chromosome: {}, objectives: { latency: 3 } }];
    expect(extractFront(items, objectives)).toEqual([[3, 0]]);
  });
});

describe("dominates", () => {
  it("requires at least one strict improvement", () => {
    expect(dominates([1, 1], [1, 1], MIN2)).toBe(false);
    expect(dominates([1, 0], [1, 1], MIN2)).toBe(true);
    expect(dominates([2, 0], [1, 1], MIN2)).toBe(false);
  });

  it("inverts comparisons for maximization objectives", () => {
    expect(dominates([2, 2], [1, 1], [false, false])).toBe(true);
    expect(dominates([1, 1], [2, 2], [false, false])).toBe(false);
  });
});

describe("coverage", () => {
  it("is 1 when A dominates every point of B and 0 for the reverse", () => {
    const A = [[0, 0]];
    const B = [
      [1, 1],
      [2, 2],
    ];
    expect(coverage(A, B, MIN2)).toBe(1);
    expect(coverage(B, A, MIN2)).toBe(0);
  });

  it("counts the dominated fraction", () => {
    const A = [[0, 3]];
    const B = [
      [1, 4], // dominated by [0,3]
      [0, 0], // not dominated
    ];
    expect(coverage(A, B, MIN2)).toBe(0.5);
  });

  it("returns 0 for an empty B", () => {
    expect(coverage([[0, 0]], [], MIN2)).toBe(0);
  });
});

describe("epsilonIndicator", () => {
  it("is <= 0 when A weakly dominates all of B", () => {
    const A = [[0, 0]];
    const B = [[1, 1]];
    expect(epsilonIndicator(A, B, MIN2)).toBeLessThanOrEqual(0);
  });

  it("equals the shift needed for A to cover B", () => {
    const A = [[2, 2]];
    const B = [[1, 1]];
    // a_i - b_i = 1 on both axes → eps = 1
    expect(epsilonIndicator(A, B, MIN2)).toBe(1);
  });

  it("is Infinity when either front is empty", () => {
    expect(epsilonIndicator([], [[1, 1]], MIN2)).toBe(Infinity);
    expect(epsilonIndicator([[1, 1]], [], MIN2)).toBe(Infinity);
  });
});

describe("igdPlus", () => {
  it("is 0 when A weakly dominates every reference point", () => {
    const A = [[0, 0]];
    const B = [
      [1, 1],
      [2, 0],
    ];
    expect(igdPlus(A, B, MIN2)).toBe(0);
  });

  it("only counts the dominated-direction distance", () => {
    const A = [[2, 0]];
    const B = [[1, 1]];
    // For min objectives: d+ = sqrt(max(2-1,0)^2 + max(0-1,0)^2) = 1
    expect(igdPlus(A, B, MIN2)).toBe(1);
  });
});

describe("spacing", () => {
  it("is 0 for fewer than two points", () => {
    expect(spacing([])).toBe(0);
    expect(spacing([[1, 1]])).toBe(0);
  });

  it("is 0 for a perfectly uniform front", () => {
    const front = [
      [0, 0],
      [1, 0],
      [2, 0],
    ];
    expect(spacing(front)).toBeCloseTo(0, 10);
  });

  it("grows when the distribution is uneven", () => {
    const uniform = [
      [0, 0],
      [1, 0],
      [2, 0],
      [3, 0],
    ];
    const clumped = [
      [0, 0],
      [0.1, 0],
      [0.2, 0],
      [3, 0],
    ];
    expect(spacing(clumped)).toBeGreaterThan(spacing(uniform));
  });
});
