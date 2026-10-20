import { describe, it, expect } from "vitest";
import { computeRanks, computeRanksWithDuplicates } from "../nonDominatedSort";

const MIN2 = [true, true];

it("matches repeated Pareto peeling for mixed goals and duplicates", () => {
  let seed = 2026;
  const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed; };
  for (const dimensions of [2, 3, 6]) {
    const minimize = Array.from({ length: dimensions }, (_, i) => i % 2 === 0);
    for (let run = 0; run < 10; run++) {
      const points = Array.from({ length: 80 }, (_, i) => ({
        id: String(i), objectives: Array.from({ length: dimensions }, () => random() % 8),
      }));
      const expected = new Map<string, number>();
      let remaining = points;
      let rank = 0;
      while (remaining.length) {
        const front = remaining.filter((p) => !remaining.some((q) =>
          q.objectives.every((v, i) => minimize[i] ? v <= p.objectives[i]! : v >= p.objectives[i]!) &&
          q.objectives.some((v, i) => v !== p.objectives[i]),
        ));
        for (const p of front) expected.set(p.id, rank);
        remaining = remaining.filter((p) => !expected.has(p.id));
        rank++;
      }
      expect(computeRanks(points, minimize)).toEqual(expected);
      expect(computeRanksWithDuplicates(points, minimize)).toEqual(expected);
    }
  }
});

describe("computeRanks", () => {
  it("puts mutually non-dominated points on the same front", () => {
    // Classic trade-off: each better on one objective
    const ranks = computeRanks(
      [
        { id: "a", objectives: [1, 4] },
        { id: "b", objectives: [2, 3] },
        { id: "c", objectives: [4, 1] },
      ],
      MIN2,
    );
    expect(ranks.get("a")).toBe(0);
    expect(ranks.get("b")).toBe(0);
    expect(ranks.get("c")).toBe(0);
  });

  it("assigns increasing ranks to successively dominated layers", () => {
    const ranks = computeRanks(
      [
        { id: "front0", objectives: [0, 0] },
        { id: "front1", objectives: [1, 1] },
        { id: "front2", objectives: [2, 2] },
      ],
      MIN2,
    );
    expect(ranks.get("front0")).toBe(0);
    expect(ranks.get("front1")).toBe(1);
    expect(ranks.get("front2")).toBe(2);
  });

  it("respects maximization objectives", () => {
    const ranks = computeRanks(
      [
        { id: "low", objectives: [1, 1] },
        { id: "high", objectives: [2, 2] },
      ],
      [false, false],
    );
    expect(ranks.get("high")).toBe(0);
    expect(ranks.get("low")).toBe(1);
  });

  it("handles mixed min/max objectives", () => {
    // minimize obj0, maximize obj1: "best" is low first, high second
    const ranks = computeRanks(
      [
        { id: "best", objectives: [1, 9] },
        { id: "worst", objectives: [5, 2] },
        { id: "tradeoff", objectives: [0, 1] },
      ],
      [true, false],
    );
    expect(ranks.get("best")).toBe(0);
    expect(ranks.get("tradeoff")).toBe(0);
    expect(ranks.get("worst")).toBe(1);
  });

  it("treats equal objective vectors as non-dominating each other", () => {
    const ranks = computeRanks(
      [
        { id: "a", objectives: [1, 1] },
        { id: "b", objectives: [1, 1] },
      ],
      MIN2,
    );
    expect(ranks.get("a")).toBe(0);
    expect(ranks.get("b")).toBe(0);
  });

  it("returns an empty map for no points", () => {
    expect(computeRanks([], MIN2).size).toBe(0);
  });
});

describe("computeRanksWithDuplicates", () => {
  it("gives duplicated objective vectors the rank of their representative", () => {
    const ranks = computeRanksWithDuplicates(
      [
        { id: "orig", objectives: [1, 1] },
        { id: "dup", objectives: [1, 1] },
        { id: "dominated", objectives: [2, 2] },
      ],
      MIN2,
    );
    expect(ranks.get("orig")).toBe(0);
    expect(ranks.get("dup")).toBe(0);
    expect(ranks.get("dominated")).toBe(1);
  });

  it("does not let duplicates push each other to a worse front", () => {
    // With naive pairwise domination, many identical points could distort
    // counts; dedup guarantees they all share one representative's rank.
    const points = Array.from({ length: 5 }, (_, i) => ({
      id: `p${i}`,
      objectives: [1, 1],
    }));
    points.push({ id: "better", objectives: [0, 0] });
    const ranks = computeRanksWithDuplicates(points, MIN2);
    expect(ranks.get("better")).toBe(0);
    for (let i = 0; i < 5; i++) expect(ranks.get(`p${i}`)).toBe(1);
  });
});
