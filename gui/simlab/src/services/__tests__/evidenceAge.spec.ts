import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { evidenceAge, parseInstant } from "../resourceLevels";

// The browser's timezone is the subject here, so it is set before anything
// reads a Date. `America/Sao_Paulo` is three hours behind UTC and is where
// this project runs — which is how R25 was found. Through `vi.stubEnv` rather
// than `process.env` so the file type-checks against the application's
// tsconfig, which has no Node types.
vi.stubEnv("TZ", "America/Sao_Paulo");

/**
 * R25, at the end of the chain.
 *
 * The API serialised `2026-10-15T15:00:00` with no offset, and
 * `new Date(...)` resolved that in the browser's timezone: a measurement one
 * minute old became three hours in the *future*, and the age came out empty
 * because the code refuses a negative one. The worker's row then showed
 * `ready` with nothing saying how old that was — which is the whole reason
 * the age exists.
 *
 * The API sends an offset now. These tests cover both forms, because that fix
 * is only true of what the API sends today: a response from a cache, a proxy
 * or an older deployment can still arrive naive.
 */

const MEASURED_AT_UTC = Date.UTC(2026, 9, 15, 15, 0, 0);

describe("parseInstant", () => {
  it("resolves a timestamp with an offset", () => {
    expect(parseInstant("2026-10-15T15:00:00+00:00")).toBe(MEASURED_AT_UTC);
    expect(parseInstant("2026-10-15T15:00:00Z")).toBe(MEASURED_AT_UTC);
  });

  it("reads one without an offset as UTC", () => {
    // The assertion that fails on any machine not on UTC if the naive string
    // is handed to `new Date` unchanged.
    expect(parseInstant("2026-10-15T15:00:00")).toBe(MEASURED_AT_UTC);
  });

  it("does not resolve a naive timestamp in the browser's timezone", () => {
    const localReading = new Date("2026-10-15T15:00:00").getTime();
    expect(localReading).not.toBe(MEASURED_AT_UTC);
    expect(parseInstant("2026-10-15T15:00:00")).not.toBe(localReading);
  });

  it("keeps a non-UTC offset as the instant it names", () => {
    expect(parseInstant("2026-10-15T12:00:00-03:00")).toBe(MEASURED_AT_UTC);
    expect(parseInstant("2026-10-15T12:00:00-0300")).toBe(MEASURED_AT_UTC);
  });
});

describe("evidenceAge with the browser three hours behind the server", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(MEASURED_AT_UTC + 60_000));
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("reads a one-minute-old measurement as one minute old", () => {
    expect(evidenceAge("2026-10-15T15:00:00+00:00")).toBe("1min ago");
  });

  it("reads a naive one the same way", () => {
    // This is the case that came out empty: interpreted locally it was three
    // hours in the future, and a negative age is refused.
    expect(evidenceAge("2026-10-15T15:00:00")).toBe("1min ago");
  });

  it("still says nothing for a measurement genuinely in the future", () => {
    // A clock skew between the server and the browser is real, and inventing
    // an age for it would be worse than saying nothing.
    expect(evidenceAge("2026-10-15T16:00:00Z")).toBe("");
  });

  it("says nothing when there is no measurement", () => {
    expect(evidenceAge(null)).toBe("");
    expect(evidenceAge(undefined)).toBe("");
  });

  it("scales from seconds to days", () => {
    const ago = (ms: number) => new Date(MEASURED_AT_UTC + 60_000 - ms).toISOString();
    expect(evidenceAge(ago(30_000))).toBe("30s ago");
    expect(evidenceAge(ago(5 * 60_000))).toBe("5min ago");
    expect(evidenceAge(ago(3 * 3_600_000))).toBe("3h ago");
    expect(evidenceAge(ago(4 * 86_400_000))).toBe("4d ago");
  });
});
