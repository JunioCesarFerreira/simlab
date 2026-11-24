import { describe, it, expect } from "vitest";
import { extractFront } from "../comparisonMetrics";
import { isPenalized, isValidEvaluation, PENALTY_THRESHOLD } from "../../types/simlab";
import type { ObjectiveItem, ParetoFrontItemDto } from "../../types/simlab";

// Mirrors pylib/evaluation_validity.py: penalties are dropped by the evidence
// stored with the vector; the magnitude rule is only for legacy documents.
describe("isPenalized with evidence", () => {
  it("keeps a large measurement that carries evidence", () => {
    expect(isPenalized([2e8, 1], { state: "complete", objective_used: true })).toBe(false);
  });

  it("drops a penalty by its record, whatever its size", () => {
    expect(isPenalized([5, 1], { state: "infeasible" })).toBe(true);
    expect(isPenalized([5, 1], { state: "complete", refused: "nan" })).toBe(true);
    expect(isPenalized([5, 1], { state: "partial", objective_used: false })).toBe(true);
    expect(isPenalized([5, 1], { valid: false })).toBe(true);
  });

  it("keeps the magnitude rule for legacy vectors, in absolute value", () => {
    expect(isPenalized([PENALTY_THRESHOLD, 0])).toBe(true);
    expect(isPenalized([-PENALTY_THRESHOLD, 0])).toBe(true);
    expect(isPenalized([1, 2], null)).toBe(false);
  });

  it("always drops non-finite vectors", () => {
    expect(isPenalized([Number.NaN, 1], { state: "complete" })).toBe(true);
  });

  it("reads a record with no fields as no evidence", () => {
    expect(isValidEvaluation(undefined)).toBe(true);
    expect(isPenalized([1e9, 1], {})).toBe(true);
  });
});

describe("extractFront uses the item's evaluation", () => {
  const objectives: ObjectiveItem[] = [
    { metric_name: "energy", goal: "min" },
    { metric_name: "latency", goal: "min" },
  ];

  it("keeps a valid large value and drops a marked penalty", () => {
    const items: ParetoFrontItemDto[] = [
      { chromosome: {}, objectives: { energy: 2e8, latency: 1 },
        evaluation: { state: "complete", objective_used: true, valid: true } },
      { chromosome: {}, objectives: { energy: 1, latency: 1 },
        evaluation: { state: "infeasible", objective_used: true, valid: false } },
    ];
    expect(extractFront(items, objectives)).toEqual([[2e8, 1]]);
  });
});
