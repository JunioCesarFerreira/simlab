/**
 * Where a simulation ran, as the panel reads it.
 *
 * The data was persisted at reservation time and never published, so the
 * chain experiment → generation → individual → simulation stopped short of
 * the machine. These pin the readings the page depends on.
 */
import { describe, expect, it } from "vitest";
import {
  busiestWorker,
  countByHost,
  describeOutcome,
  emptyMessage,
  formatMeanDuration,
  formatShare,
  placementLabel,
  ranInTheSamePlace,
  segmentWidth,
  workerCount,
} from "../placement";
import type {
  PlacementDistributionDto,
  PlacementDto,
  PlacementWorkerDto,
  SimulationDto,
} from "../../types/simlab";

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

/* ── The experiment-wide distribution ─────────────────────────────────── */

function worker(overrides: Partial<PlacementWorkerDto> = {}): PlacementWorkerDto {
  return {
    resource_name: "cooja1",
    total: 4,
    done: 4,
    failed: 0,
    active: 0,
    share: 50,
    ...overrides,
  };
}

function distribution(overrides: Partial<PlacementDistributionDto> = {}): PlacementDistributionDto {
  return {
    total: 8,
    placed: 8,
    unplaced: 0,
    hosts: [
      {
        host_name: "local",
        total: 6,
        done: 6,
        failed: 0,
        active: 0,
        share: 75,
        workers: [
          worker({ total: 4, done: 4, share: 50 }),
          worker({ resource_name: "cooja2", total: 2, done: 2, share: 25 }),
        ],
      },
      {
        host_name: "vm8",
        total: 2,
        done: 2,
        failed: 0,
        active: 0,
        share: 25,
        workers: [worker({ resource_name: "vm8-cooja1", total: 2, done: 2, share: 25 })],
      },
    ],
    ...overrides,
  };
}

describe("workerCount", () => {
  it("counts the containers across every host", () => {
    expect(workerCount(distribution())).toBe(3);
  });

  it("is zero without a distribution", () => {
    expect(workerCount(null)).toBe(0);
    expect(workerCount(distribution({ hosts: [] }))).toBe(0);
  });
});

describe("busiestWorker", () => {
  it("is the largest container count, which the bars are scaled against", () => {
    // Not the total: with twelve workers every bar would be a sliver, and the
    // question is which of them carried more.
    expect(busiestWorker(distribution())).toBe(4);
  });

  it("is never zero, so a caller can divide by it", () => {
    expect(busiestWorker(null)).toBe(1);
    expect(busiestWorker(distribution({ hosts: [] }))).toBe(1);
  });
});

describe("segmentWidth", () => {
  it("renders the busiest container as a full bar", () => {
    expect(segmentWidth(4, 4)).toBe(100);
  });

  it("scales the others against it", () => {
    expect(segmentWidth(1, 4)).toBe(25);
  });

  it("does not divide by zero", () => {
    expect(segmentWidth(3, 0)).toBe(0);
  });
});

describe("formatShare", () => {
  it("shows one decimal", () => {
    expect(formatShare(33.333)).toBe("33.3%");
  });

  it("does not round a container that ran something down to nothing", () => {
    // 1 of 4000 simulations is 0.025%, and "0.0%" reads as "this one did
    // nothing" — the opposite of what it did.
    expect(formatShare(0.025)).toBe("<0.1%");
  });

  it("still shows a true zero as zero", () => {
    expect(formatShare(0)).toBe("0.0%");
  });
});

describe("describeOutcome", () => {
  it("always says how many finished, including none", () => {
    // A container that took work and returned nothing is the case a single
    // count cannot show.
    expect(describeOutcome(worker({ total: 3, done: 0, failed: 3 }))).toBe("0 done, 3 failed");
  });

  it("leaves out what did not happen", () => {
    expect(describeOutcome(worker({ done: 4 }))).toBe("4 done");
  });

  it("names work still in flight", () => {
    expect(describeOutcome(worker({ done: 1, failed: 1, active: 2 }))).toBe(
      "1 done, 1 failed, 2 running",
    );
  });
});

describe("formatMeanDuration", () => {
  it("is a dash when nothing finished, never zero", () => {
    // Zero would read as the fastest worker.
    expect(formatMeanDuration(null)).toBe("—");
    expect(formatMeanDuration(undefined)).toBe("—");
  });

  it("keeps tenths below a minute", () => {
    expect(formatMeanDuration(1.24)).toBe("1.2 s");
    expect(formatMeanDuration(59.9)).toBe("59.9 s");
  });

  it("reads minutes and hours past that", () => {
    expect(formatMeanDuration(90.4)).toBe("1 min 30 s");
    expect(formatMeanDuration(3725)).toBe("1 h 2 min");
  });
});

describe("emptyMessage", () => {
  it("says nothing when there are bars to show", () => {
    expect(emptyMessage(distribution())).toBe("");
  });

  it("distinguishes queued work from work that never ran", () => {
    expect(emptyMessage(distribution({ placed: 0, unplaced: 8, hosts: [] }))).toContain(
      "8 waiting",
    );
    expect(emptyMessage(distribution({ total: 0, placed: 0, unplaced: 0, hosts: [] }))).toBe(
      "No simulation of this experiment ran on a container.",
    );
  });

  it("distinguishes both from having no record at all", () => {
    expect(emptyMessage(null)).toBe("No record of where this experiment ran.");
  });
});
