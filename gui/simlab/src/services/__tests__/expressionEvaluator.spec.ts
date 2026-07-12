import { describe, it, expect } from "vitest";
import { evalExpr, sampleCustomSegment, validateExpr } from "../expressionEvaluator";

describe("evalExpr", () => {
  it("evaluates plain arithmetic in t", () => {
    const r = evalExpr("2 * t + 1", 0.5);
    expect(r).toEqual({ ok: true, value: 2 });
  });

  it("translates NumPy syntax to Math", () => {
    expect(evalExpr("np.cos(0)", 0)).toEqual({ ok: true, value: 1 });
    expect(evalExpr("np.sin(np.pi / 2)", 0)).toEqual({ ok: true, value: 1 });
    expect(evalExpr("np.sqrt(t)", 4)).toEqual({ ok: true, value: 2 });
  });

  it("rejects non-finite results", () => {
    const r = evalExpr("1 / t", 0);
    expect(r.ok).toBe(false);
  });

  it("returns an error for invalid syntax instead of throwing", () => {
    const r = evalExpr("np.cos(", 0);
    expect(r.ok).toBe(false);
  });
});

describe("sampleCustomSegment", () => {
  it("samples n+1 points from t=0 to t=1", () => {
    const pts = sampleCustomSegment("t", "2 * t", 4);
    expect(pts).toEqual([
      [0, 0],
      [0.25, 0.5],
      [0.5, 1],
      [0.75, 1.5],
      [1, 2],
    ]);
  });

  it("returns null when any evaluation fails", () => {
    expect(sampleCustomSegment("1 / (t - 0.5)", "t", 4)).toBeNull();
  });
});

describe("validateExpr", () => {
  it("accepts valid expressions", () => {
    expect(validateExpr("np.cos(2 * np.pi * t)")).toBeNull();
  });

  it("rejects empty and broken expressions with a message", () => {
    expect(validateExpr("   ")).toBe("Expression is empty");
    expect(validateExpr("t +")).toBeTruthy();
  });
});
