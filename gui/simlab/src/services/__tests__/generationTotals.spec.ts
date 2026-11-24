import { describe, it, expect } from "vitest";
import { plannedGenerations, plannedSimulations } from "../generationTotals";

describe("planned totals", () => {
  it("count generation 0", () => {
    // 10 evolution steps are 11 generations: the page read 11 / 10.
    expect(plannedGenerations({ number_of_generations: 10 }, 3)).toBe(11);
    expect(plannedSimulations({ population_size: 20, number_of_generations: 10 }, 2)).toBe(20 * 11 * 2);
  });

  it("fall back to what exists when the strategy plans no generations", () => {
    expect(plannedGenerations({}, 1)).toBe(1);
    expect(plannedGenerations(undefined, 0)).toBe(0);
    expect(plannedSimulations({}, 2)).toBeNull();
  });

  it("accept zero evolution steps", () => {
    expect(plannedGenerations({ number_of_generations: 0 }, 0)).toBe(1);
    expect(plannedSimulations({ population_size: 4, number_of_generations: 0 }, 1)).toBe(4);
  });
});
