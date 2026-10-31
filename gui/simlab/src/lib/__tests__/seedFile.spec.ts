import { describe, expect, it } from "vitest";
import { parseSeedFile, serializeSeedFile } from "../seedFile";

describe("parseSeedFile", () => {
  it("reads the wizard's format", () => {
    expect(parseSeedFile('{ "seeds": [3, 1, 2] }')).toEqual({ ok: true, seeds: [3, 1, 2] });
  });

  it("reads a bare list", () => {
    expect(parseSeedFile("[7, 8]")).toEqual({ ok: true, seeds: [7, 8] });
  });

  it("keeps repeated values: whether they are allowed is the caller's decision", () => {
    expect(parseSeedFile("[7, 7]")).toEqual({ ok: true, seeds: [7, 7] });
  });

  it.each([
    ["not json", "Invalid JSON file"],
    ["{}", 'Invalid file — expected { "seeds": [...] }'],
    ["null", 'Invalid file — expected { "seeds": [...] }'],
    ["[]", 'Invalid file — expected { "seeds": [...] }'],
    ['["a"]', "Invalid file — seeds must be integers"],
  ])("refuses %s", (text, error) => {
    expect(parseSeedFile(text)).toEqual({ ok: false, error });
  });

  it("reads back what it writes", () => {
    expect(parseSeedFile(serializeSeedFile([1, 2]))).toEqual({ ok: true, seeds: [1, 2] });
  });
});
