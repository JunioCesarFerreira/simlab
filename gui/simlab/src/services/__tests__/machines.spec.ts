/**
 * What an operator types as a machine's limit. Blank is "no limit", which
 * is a value, not an error.
 */
import { describe, expect, it } from "vitest";
import { parseLimit } from "../machines";

describe("parseLimit", () => {
  it("reads a whole number", () => {
    expect(parseLimit(" 5 ")).toEqual({ ok: true, value: 5 });
  });

  it("reads blank as no limit", () => {
    expect(parseLimit("")).toEqual({ ok: true, value: null });
    expect(parseLimit("   ")).toEqual({ ok: true, value: null });
  });

  it("refuses zero, which would stop the machine entirely", () => {
    expect(parseLimit("0").ok).toBe(false);
  });

  it("refuses what the API would", () => {
    for (const bad of ["2.5", "-1", "five", "1e3"]) expect(parseLimit(bad).ok).toBe(false);
  });
});
