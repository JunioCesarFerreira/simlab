/**
 * Where a simulation ran, as the panel reads it.
 *
 * The data was persisted at reservation time and never published, so the
 * chain experiment → generation → individual → simulation stopped short of
 * the machine. These pin the readings the page depends on.
 */
import { describe, expect, it } from "vitest";
import { countByHost, placementLabel, ranInTheSamePlace } from "../placement";
import type { PlacementDto, SimulationDto } from "../../types/simlab";

function placement(overrides: Partial<PlacementDto> = {}): PlacementDto {
  return {
    host_name: "vm8",
    resource_name: "vm8-cooja2",
    container_id: "abc123",
    ...overrides,
  };
}

function simulation(p?: PlacementDto | null): SimulationDto {
  return { placement: p } as unknown as SimulationDto;
}

describe("placementLabel", () => {
  it("reads host and worker", () => {
    expect(placementLabel(placement())).toBe("vm8/vm8-cooja2");
  });

  it("says nothing for a simulation that was never placed", () => {
    // Different from one that ran somewhere unknown.
    expect(placementLabel(null)).toBe("");
    expect(placementLabel(undefined)).toBe("");
  });

  it("marks a missing half rather than hiding it", () => {
    expect(placementLabel(placement({ resource_name: "" }))).toBe("vm8/?");
  });
});

describe("ranInTheSamePlace", () => {
  it("the container decides when both have one", () => {
    // A worker's *name* can be reused by a rebuilt container, and then two
    // runs that look alike came from different machines.
    const a = placement({ container_id: "abc" });
    const b = placement({ container_id: "def" });

    expect(ranInTheSamePlace(a, b)).toBe(false);
  });

  it("the same container is the same place even if renamed", () => {
    const a = placement({ container_id: "abc", resource_name: "old" });
    const b = placement({ container_id: "abc", resource_name: "new" });

    expect(ranInTheSamePlace(a, b)).toBe(true);
  });

  it("falls back to host and worker for an older record", () => {
    const a = placement({ container_id: null });
    const b = placement({ container_id: null });

    expect(ranInTheSamePlace(a, b)).toBe(true);
    expect(ranInTheSamePlace(a, placement({ container_id: null, host_name: "vm7" }))).toBe(false);
  });

  it("an unplaced run is nowhere, not somewhere", () => {
    expect(ranInTheSamePlace(placement(), null)).toBe(false);
    expect(ranInTheSamePlace(null, null)).toBe(false);
  });
});

describe("countByHost", () => {
  it("counts the runs each host took", () => {
    // What makes the distribution of work auditable from the results.
    const counts = countByHost([
      simulation(placement({ host_name: "local" })),
      simulation(placement({ host_name: "local" })),
      simulation(placement({ host_name: "vm8" })),
    ]);

    expect(counts.get("local")).toBe(2);
    expect(counts.get("vm8")).toBe(1);
  });

  it("leaves out what never ran", () => {
    const counts = countByHost([simulation(null), simulation(placement())]);

    expect([...counts.keys()]).toEqual(["vm8"]);
  });
});
