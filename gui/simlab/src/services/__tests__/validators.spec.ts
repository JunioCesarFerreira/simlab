import { describe, it, expect } from "vitest";
import { validateProblem, validateChromosome } from "../validators";
import type { ProblemDraft } from "../../types/problem";

function baseDraft(overrides: Partial<ProblemDraft> = {}): ProblemDraft {
  return {
    name: "problem2",
    radiusOfReach: 100,
    radiusOfInter: 200,
    radiusOfCover: 90,
    kRequired: 1,
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

describe("validateProblem", () => {
  it("accepts a minimal valid draft", () => {
    expect(validateProblem(baseDraft())).toEqual([]);
  });

  it("requires a name and positive radii", () => {
    const errors = validateProblem(
      baseDraft({ name: "  ", radiusOfReach: 0, radiusOfInter: -1 }),
    );
    const fields = errors.map((e) => e.field);
    expect(fields).toContain("name");
    expect(fields).toContain("radius_of_reach");
    expect(fields).toContain("radius_of_inter");
  });

  it("rejects an inverted region", () => {
    const errors = validateProblem(baseDraft({ region: [10, 10, -10, -10], sink: null }));
    expect(errors.filter((e) => e.field === "region")).toHaveLength(2);
  });

  it("requires the sink and keeps it inside the region", () => {
    expect(validateProblem(baseDraft({ sink: null })).map((e) => e.field)).toContain("sink");
    expect(
      validateProblem(baseDraft({ sink: { x: 999, y: 0 } })).map((e) => e.field),
    ).toContain("sink");
  });

  it("flags candidates outside the region", () => {
    const errors = validateProblem(
      baseDraft({ candidates: [{ id: "c1", x: 500, y: 0 }] }),
    );
    expect(errors.map((e) => e.field)).toContain("candidates");
  });

  it("validates mobile node fields", () => {
    const errors = validateProblem(
      baseDraft({
        mobileNodes: [
          {
            id: "m1",
            name: "",
            sourceCode: "",
            speed: 0,
            timeStep: 0,
            isClosed: false,
            isRoundTrip: false,
            segments: [],
          },
        ],
      }),
    );
    // name, source, speed, timeStep, segments
    expect(errors.filter((e) => e.field === "mobile_nodes")).toHaveLength(5);
  });
});

describe("validateChromosome", () => {
  it("passes when there is no chromosome", () => {
    expect(validateChromosome(baseDraft())).toEqual([]);
  });

  it("rejects a chromosome whose kind mismatches the problem", () => {
    const errors = validateChromosome(
      baseDraft({
        name: "problem2",
        chromosome: { kind: "problem1", macProtocol: "csma", relays: [] },
      }),
    );
    expect(errors[0]?.field).toBe("chromosome");
  });

  it("problem1: relay count must equal numSensors and relays stay in region", () => {
    const errors = validateChromosome(
      baseDraft({
        name: "problem1",
        numSensors: 2,
        chromosome: {
          kind: "problem1",
          macProtocol: "csma",
          relays: [{ id: "r1", x: 999, y: 0 }],
        },
      }),
    );
    const fields = errors.map((e) => e.field);
    expect(fields).toContain("chromosome.relays");
    // missing relay + out-of-region relay
    expect(errors).toHaveLength(2);
  });

  it("problem2: mask must align with candidates and be binary", () => {
    const errors = validateChromosome(
      baseDraft({
        candidates: [{ id: "c1", x: 0, y: 0 }],
        chromosome: { kind: "problem2", macProtocol: "csma", mask: [0, 2] },
      }),
    );
    const fields = errors.map((e) => e.field);
    expect(fields.filter((f) => f === "chromosome.mask")).toHaveLength(2);
  });

  it("problem4: route indices bounded and sojourn times aligned and non-negative", () => {
    const errors = validateChromosome(
      baseDraft({
        name: "problem4",
        candidates: [{ id: "c1", x: 0, y: 0 }],
        chromosome: {
          kind: "problem4",
          macProtocol: "csma",
          route: [0, 5],
          sojournTimes: [-1],
        },
      }),
    );
    const fields = errors.map((e) => e.field);
    expect(fields).toContain("chromosome.route");
    expect(fields).toContain("chromosome.sojourn_times");
  });
});
