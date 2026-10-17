import { describe, it, expect } from "vitest";
import {
  completenessDetail,
  completenessLabel,
  completenessTone,
  thinMetrics,
} from "../completeness";
import type { CompletenessDto } from "../../types/simlab";

/**
 * R19, at the end of the chain. The counts are computed in the persistence
 * library, decided by the engine's policy and published by the API; this is
 * the last step, and the one an operator actually reads.
 *
 * The distinction these tests protect is "not recorded" versus "complete".
 * An individual evaluated before completeness existed has no record, and
 * showing it as complete would be a claim about data nobody wrote down.
 */

const complete: CompletenessDto = {
  state: "complete",
  expected: 2,
  observed: 2,
  per_metric: { latency: 2, energy: 2 },
  attempts: 2,
  objective_used: true,
};

const partial: CompletenessDto = {
  state: "partial",
  expected: 2,
  observed: 1,
  failed: 1,
  missing_seeds: [5343],
  failed_seeds: [5343],
  per_metric: { latency: 1 },
  attempts: 2,
  objective_used: false,
  policy: "repeat missing replicas up to 1 time(s) per generation, then worst",
};

describe("completenessLabel", () => {
  it("names the replicas behind a complete evaluation", () => {
    expect(completenessLabel(complete)).toBe("complete (2/2 replicas)");
  });

  it("says incomplete rather than giving a state name", () => {
    expect(completenessLabel(partial)).toBe("incomplete (1/2 replicas)");
  });

  it("distinguishes not recorded from complete", () => {
    expect(completenessLabel(null)).toBe("not recorded");
    expect(completenessLabel(undefined)).toBe("not recorded");
  });

  it("says unknown when the generation never recorded its plan", () => {
    expect(completenessLabel({ state: "unknown", observed: 1 })).toContain("unknown");
  });

  it("reads a pending evaluation as still running", () => {
    expect(completenessLabel({ state: "pending", expected: 2, observed: 1 })).toBe(
      "still running (1/2 replicas)",
    );
  });
});

describe("completenessTone", () => {
  it("is ok only for a complete evaluation", () => {
    expect(completenessTone(complete)).toBe("ok");
  });

  it("warns for partial and pending", () => {
    expect(completenessTone(partial)).toBe("warn");
    expect(completenessTone({ state: "pending" })).toBe("warn");
  });

  it("is bad when nothing finished", () => {
    expect(completenessTone({ state: "invalid" })).toBe("bad");
  });

  it("is unknown for both no record and an unknown state", () => {
    expect(completenessTone(null)).toBe("unknown");
    expect(completenessTone({ state: "unknown" })).toBe("unknown");
  });
});

describe("thinMetrics", () => {
  it("names a metric that fewer replicas carried", () => {
    // Both replicas finished; one of them had no energy reading. A single
    // "2 of 2" cannot express that.
    expect(
      thinMetrics({ state: "partial", expected: 2, per_metric: { latency: 2, energy: 1 } }),
    ).toEqual(["energy: 1/2"]);
  });

  it("is empty when every metric is there", () => {
    expect(thinMetrics(complete)).toEqual([]);
  });

  it("is empty without the counts to compare", () => {
    expect(thinMetrics({ state: "complete" })).toEqual([]);
  });
});

describe("completenessDetail", () => {
  it("names the seed to look at", () => {
    expect(completenessDetail(partial)).toContain("missing seeds: 5343");
  });

  it("says when the algorithm was not given the vector", () => {
    expect(completenessDetail(partial)).toContain(
      "the algorithm was given the worst objective for this individual",
    );
  });

  it("says when a replica was repeated", () => {
    expect(
      completenessDetail({ state: "complete", expected: 2, observed: 2, attempts: 3 }),
    ).toContain("3 runs for 2 replicas (repeated)");
  });

  it("separates a replica that was never created from one that failed", () => {
    expect(completenessDetail({ state: "partial", expected: 2, observed: 1, absent: 1 })).toContain(
      "1 replica(s) were never created",
    );
  });

  it("names the replicas that were never created", () => {
    // The hole no repeat can fill, so the reader needs to know which replica
    // was decided without ever running.
    expect(
      completenessDetail({
        state: "partial",
        expected: 3,
        observed: 1,
        absent: 2,
        absent_seeds: [222, 333],
      }),
    ).toContain("2 replica(s) were never created (seeds 222, 333)");
  });

  it("shows the reason the engine recorded", () => {
    // `partial` with every planned seed finished: only the reason explains it.
    expect(
      completenessDetail({
        state: "partial",
        expected: 2,
        observed: 2,
        thin_metrics: ["energy"],
        reason: "finished replicas are missing energy (nothing to repeat)",
      }),
    ).toContain("finished replicas are missing energy (nothing to repeat)");
  });

  it("explains an absent record instead of showing nothing", () => {
    expect(completenessDetail(null)).toEqual([
      "This evaluation was made before completeness was recorded.",
    ]);
  });

  it("says nothing extra about a clean complete evaluation", () => {
    expect(completenessDetail(complete)).toEqual([]);
  });
});

describe("the states that are not about replicas", () => {
  it("names an analytical evaluation instead of calling it unknown", () => {
    // `unknown` means "the plan was not recorded". An analytical problem had
    // no plan to record, which is a different thing.
    const made: CompletenessDto = { state: "analytical", expected: 0, observed: 0 };

    expect(completenessLabel(made)).toBe("evaluated analytically");
    expect(completenessTone(made)).toBe("ok");
  });

  it("names an individual rejected by a constraint", () => {
    const made: CompletenessDto = { state: "infeasible", expected: 0, observed: 0 };

    expect(completenessLabel(made)).toBe("rejected by a constraint");
    // Not an error: a constrained search rejecting candidates is normal.
    expect(completenessTone(made)).toBe("unknown");
  });

  it("shows the reason when there is one", () => {
    expect(
      completenessDetail({
        state: "infeasible",
        reason: "the individual violates a constraint; no simulation was run",
      }),
    ).toContain("the individual violates a constraint; no simulation was run");
  });
});

describe("a carried-over evaluation", () => {
  it("says so, because it has no replicas of its own here", () => {
    expect(completenessDetail({ ...complete, reused: true, source: "genome_cache" })).toContain(
      "carried over from an earlier evaluation of the same genome",
    );
  });

  it("keeps the numbers of the evaluation it came from", () => {
    expect(completenessLabel({ ...complete, reused: true })).toBe("complete (2/2 replicas)");
  });

  it("keeps a refusal visible", () => {
    const carried: CompletenessDto = { ...partial, reused: true };

    expect(completenessTone(carried)).toBe("warn");
    expect(completenessDetail(carried)).toContain(
      "the algorithm was given the worst objective for this individual",
    );
  });
});
