import { describe, it, expect } from "vitest";
import { exportProblem } from "../exportProblemJson";
import { importProblemJson } from "../importProblemJson";
import { exportChromosome } from "../exportChromosomeJson";
import { importChromosomeJson } from "../importChromosomeJson";
import { validateProblem } from "../validators";
import { DEFAULT_P4 } from "../../types/problem";
import type { ProblemDraft } from "../../types/problem";
// The same file is run by the engine (mo-engine/tests/test_p4_gui_export.py).
import fixture from "./fixtures/p4-export.json";

function draft(overrides: Partial<ProblemDraft> = {}): ProblemDraft {
  return {
    name: "problem4",
    radiusOfReach: 50,
    radiusOfInter: 60,
    radiusOfCover: 90,
    kRequired: 1,
    minCoveragePercentage: 100,
    region: [-100, -100, 100, 100],
    sink: { x: 0, y: 0 },
    candidates: [{ id: "a", x: 40, y: 0 }, { id: "b", x: -40, y: 0 }],
    targets: [{ id: "m", x: 60, y: 0 }, { id: "n", x: -80, y: 10 }],
    numSensors: 1,
    mobileNodes: [],
    chromosome: null,
    p4: { ...DEFAULT_P4 },
    ...overrides,
  };
}

describe("P4 export", () => {
  it("writes the shape the engine reads", () => {
    expect(exportProblem(draft())).toEqual({ problem: fixture.problem });
  });

  it("makes the sink the base and sojourn 0, and gives each id its index", () => {
    const { problem } = exportProblem(draft());
    expect(problem.sojourns?.[0]?.position).toEqual([0, 0]);
    expect(problem.sojourns?.map(s => s.id)).toEqual([0, 1, 2]);
  });

  it("drops mobile nodes drawn for another problem", () => {
    const withNode = draft({
      mobileNodes: [{ id: "x", name: "n", sourceCode: "node.c", speed: 1, timeStep: 1,
        isClosed: false, isRoundTrip: false, segments: [] }],
    });
    expect(exportProblem(withNode).problem.mobile_nodes).toEqual([]);
  });

  it("survives an export → import round-trip", () => {
    const result = importProblemJson(JSON.stringify(exportProblem(draft({
      p4: { ...DEFAULT_P4, speed: 3, tauMin: 2, tauMax: 9, maxRouteLength: 7 },
    }))));
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.draft.candidates.map(c => [c.x, c.y])).toEqual([[40, 0], [-40, 0]]);
    expect(result.draft.targets.map(t => [t.x, t.y])).toEqual([[60, 0], [-80, 10]]);
    expect(result.draft.sink).toEqual({ x: 0, y: 0 });
    expect(result.draft.p4).toMatchObject({ speed: 3, tauMin: 2, tauMax: 9, maxRouteLength: 7 });
  });
});

describe("P4 chromosome", () => {
  it("puts the base at both ends and numbers the stops from 1", () => {
    const exported = exportChromosome({ kind: "problem4", macProtocol: "csma", route: [0, 1], sojournTimes: [5, 5] });
    expect(exported).toEqual(fixture.chromosome);
  });

  it("comes back as it left", () => {
    const back = importChromosomeJson(JSON.stringify(fixture.chromosome));
    expect(back).toEqual({ ok: true, chromosome: { kind: "problem4", macProtocol: "csma", route: [0, 1], sojournTimes: [5, 5] } });
  });
});

describe("P4 validation", () => {
  it("accepts the draft", () => {
    expect(validateProblem(draft())).toEqual([]);
  });

  it.each([
    [{ targets: [] }, "targets"],
    [{ candidates: [] }, "candidates"],
    [{ p4: { ...DEFAULT_P4, speed: 0 } }, "p4"],
    [{ p4: { ...DEFAULT_P4, tauMin: 5, tauMax: 1 } }, "p4"],
    [{ p4: { ...DEFAULT_P4, maxRouteLength: 1 } }, "p4"],
  ] as const)("refuses %o", (overrides, field) => {
    expect(validateProblem(draft(overrides as Partial<ProblemDraft>)).map(e => e.field)).toContain(field);
  });
});

describe("P3 g", () => {
  const p3 = (g?: number) => draft({ name: "problem3", gRequired: g, p4: undefined,
    targets: [{ id: "t", x: 10, y: 0 }] });

  it("is exported, 1 when unset", () => {
    expect(exportProblem(p3(2)).problem.g_required).toBe(2);
    expect(exportProblem(p3()).problem.g_required).toBe(1);
  });

  it("comes back on import", () => {
    const result = importProblemJson(JSON.stringify(exportProblem(p3(3))));
    expect(result.ok && result.draft.gRequired).toBe(3);
  });

  it("must be a whole number of at least 1", () => {
    expect(validateProblem(p3(0)).map(e => e.field)).toContain("g_required");
    expect(validateProblem(p3(1.5)).map(e => e.field)).toContain("g_required");
    expect(validateProblem(p3(2))).toEqual([]);
  });
});
