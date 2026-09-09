import { describe, it, expect } from "vitest";
import { exportProblem } from "../exportProblemJson";
import { importProblemJson } from "../importProblemJson";
import type { ProblemDraft } from "../../types/problem";

function baseDraft(overrides: Partial<ProblemDraft> = {}): ProblemDraft {
  return {
    name: "problem2",
    radiusOfReach: 100,
    radiusOfInter: 200,
    radiusOfCover: 90,
    kRequired: 1,
    minCoveragePercentage: 100,
    region: [-100, -100, 100, 100],
    sink: { x: 0, y: 0 },
    candidates: [],
    targets: [],
    numSensors: 1,
    mobileNodes: [],
    chromosome: null,
    ...overrides,
  };
}

describe("alpha (min_coverage_percentage) round-trip", () => {
  it("is exported for the coverage-constrained problems", () => {
    for (const name of ["problem1", "problem2"]) {
      const { problem } = exportProblem(baseDraft({ name, minCoveragePercentage: 85 }));
      expect(problem.min_coverage_percentage).toBe(85);
    }
  });

  it("is omitted for problems whose adapter does not read it", () => {
    for (const name of ["problem3", "problem4"]) {
      const { problem } = exportProblem(baseDraft({ name, minCoveragePercentage: 85 }));
      expect(problem.min_coverage_percentage).toBeUndefined();
    }
  });

  it("survives an export → import round-trip", () => {
    const exported = exportProblem(baseDraft({ minCoveragePercentage: 72 }));
    const result = importProblemJson(JSON.stringify(exported));
    expect(result.ok).toBe(true);
    if (result.ok) expect(result.draft.minCoveragePercentage).toBe(72);
  });

  it("defaults to 100% when a legacy problem file omits it", () => {
    const legacy = {
      problem: {
        name: "problem2",
        radius_of_reach: 100,
        radius_of_inter: 200,
        region: [-100, -100, 100, 100],
        sink: [0, 0],
        candidates: [],
        mobile_nodes: [],
      },
    };
    const result = importProblemJson(JSON.stringify(legacy));
    expect(result.ok).toBe(true);
    if (result.ok) expect(result.draft.minCoveragePercentage).toBe(100);
  });
});
