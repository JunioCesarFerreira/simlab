/**
 * The bar that says how a host's containers are being used.
 *
 * The reported bug: switching a worker back on left it painted as
 * *disabled*. That was decided in the backend partition — the administrative
 * bucket was keyed on `status == disabled` as well as on the `enabled` flag,
 * and the status only changes when the next health cycle measures the
 * container. These tests pin the readings the bar depends on, including the
 * counts the fixed backend produces for each step of a toggle.
 */
import { describe, expect, it } from "vitest";
import {
  occupancyIsConsistent,
  occupancySegments,
  occupancySummary,
  occupancyTotal,
} from "../occupancy";
import type { OccupancyCounts } from "../occupancy";

function counts(over: Partial<OccupancyCounts> = {}): OccupancyCounts {
  return { free: 0, active: 0, unavailable: 0, disabled: 0, ...over };
}

describe("the three steps of a toggle, as the API reports them", () => {
  // Taken from the live `/infra/overview` of an 8-container host.
  const idle = counts({ resourceTotal: 8, free: 8 });
  const off = counts({ resourceTotal: 8, free: 7, disabled: 1 });
  const backOn = counts({ resourceTotal: 8, free: 7, unavailable: 1 });

  it("shows every container free to begin with", () => {
    expect(occupancySummary(idle)).toBe("8 free");
  });

  it("moves one into disabled when it is switched off", () => {
    expect(occupancySummary(off)).toBe("7 free, 1 disabled");
  });

  it("does not leave it there once it is switched back on", () => {
    // The bug: this used to read "7 free, 1 disabled" while the toggle next
    // to it said the worker was enabled.
    expect(occupancySummary(backOn)).toBe("7 free, 1 unavailable");
    expect(occupancySegments(backOn).map((s) => s.kind)).not.toContain("disabled");
  });

  it("keeps the bar full at every step", () => {
    for (const step of [idle, off, backOn]) {
      const drawn = occupancySegments(step).reduce((n, s) => n + s.percent, 0);
      expect(drawn).toBeCloseTo(100);
    }
  });
});

describe("occupancyTotal", () => {
  it("is the inventory the API sent", () => {
    expect(occupancyTotal(counts({ resourceTotal: 8, free: 7, disabled: 1 }))).toBe(8);
  });

  it("falls back to the sum for an API that does not send it", () => {
    expect(occupancyTotal(counts({ free: 7, disabled: 1 }))).toBe(8);
  });

  it("is zero for a host with no containers", () => {
    expect(occupancyTotal(counts({ resourceTotal: 0 }))).toBe(0);
  });
});

describe("occupancySegments", () => {
  it("drops the categories that are empty", () => {
    expect(occupancySegments(counts({ resourceTotal: 4, free: 4 })).map((s) => s.kind)).toEqual([
      "free",
    ]);
  });

  it("reads in use, then idle, then what cannot be used", () => {
    expect(
      occupancySegments(
        counts({ resourceTotal: 4, active: 1, free: 1, unavailable: 1, disabled: 1 }),
      ).map((s) => s.kind),
    ).toEqual(["active", "free", "unavailable", "disabled"]);
  });

  it("never draws more than a full bar, even if the counts disagree", () => {
    // Not a state the fixed backend produces; the guard is here because a bar
    // that overflows its own container is worse than a short one, and the
    // component cannot tell which number is wrong.
    const broken = counts({ resourceTotal: 2, active: 2, disabled: 2 });

    const drawn = occupancySegments(broken).reduce((n, s) => n + s.percent, 0);

    expect(drawn).toBeCloseTo(100);
  });

  it("draws nothing for a host with no containers", () => {
    expect(occupancySegments(counts({ resourceTotal: 0 }))).toEqual([]);
  });
});

describe("occupancyIsConsistent", () => {
  it("is true when the categories add up to the inventory", () => {
    expect(occupancyIsConsistent(counts({ resourceTotal: 8, free: 7, unavailable: 1 }))).toBe(true);
  });

  it("is false when a container was counted twice", () => {
    // The shape of the earlier defect: busy *and* disabled for draining
    // counted in both buckets.
    expect(occupancyIsConsistent(counts({ resourceTotal: 1, active: 1, disabled: 1 }))).toBe(false);
  });

  it("is false when a container fell out of every category", () => {
    expect(occupancyIsConsistent(counts({ resourceTotal: 3, free: 2 }))).toBe(false);
  });

  it("cannot judge an API that sends no inventory", () => {
    expect(occupancyIsConsistent(counts({ free: 2 }))).toBe(true);
  });
});

describe("occupancySummary", () => {
  it("says so when there is nothing on the host", () => {
    expect(occupancySummary(counts({ resourceTotal: 0 }))).toBe("No containers on this host");
  });

  it("adds draining as an attribute of the workers in use", () => {
    // Not a category of its own: the two draining containers are already
    // counted under "in use", and counting them again is what made a bar
    // report more containers than the host has.
    expect(occupancySummary(counts({ resourceTotal: 4, active: 3, free: 1, draining: 2 }))).toBe(
      "3 in use, 1 free, 2 of them draining",
    );
  });
});
