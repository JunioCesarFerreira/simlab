import { describe, expect, it } from "vitest";
import type { HvGdData } from "../../api/metrics";
import { metricGenerationLabels, populationCaption, populationLabel } from "../metricPopulation";

const mixed = {
  generations: [0, 2, 5],
  population: "survivors",
  population_source: "mixed",
  population_sources: ["offspring", "survivors", "offspring"],
} as HvGdData;

describe("measured population labels", () => {
  it("identifies the actual generation indices with offspring fallback", () => {
    expect(populationCaption(mixed)).toContain("offspring used for generations 0, 5");
    expect(populationCaption(mixed)).toContain("mixed series");
    expect(populationLabel(mixed.population_source)).toBe("Mixed survivors / offspring");
    expect(metricGenerationLabels(mixed)).toEqual([
      "Gen 0 (offspring)", "Gen 2 (survivors)", "Gen 5 (offspring)",
    ]);
  });

  it("explains a complete fallback without claiming the run predates survivor support", () => {
    const data = { ...mixed, population_source: "offspring" } as HvGdData;
    expect(populationCaption(data)).toContain("survivor sets are unavailable");
    expect(populationCaption(data)).not.toContain("mixed");
  });

  it.each(["survivors", "offspring", "archive"] as const)(
    "keeps homogeneous %s labels compatible with an older API", (source) => {
      const data = { generations: [3], population: source, population_source: source } as HvGdData;
      expect(populationCaption(data)).toBe(`Measured on the ${source}.`);
      expect(metricGenerationLabels(data)).toEqual(["Gen 3"]);
    },
  );
});
