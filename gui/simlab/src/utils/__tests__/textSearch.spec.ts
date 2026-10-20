import { describe, it, expect } from "vitest";
import { normalizeText, matchesQuery } from "../textSearch";

describe("normalizeText", () => {
  it("strips diacritics and lowercases", () => {
    expect(normalizeText("Estimativa de População")).toBe("estimativa de populacao");
    expect(normalizeText("Variações")).toBe("variacoes");
  });
});

describe("matchesQuery", () => {
  it("matches an empty or blank query", () => {
    expect(matchesQuery("anything", "")).toBe(true);
    expect(matchesQuery("anything", "   ")).toBe(true);
  });

  it("matches a substring regardless of case", () => {
    expect(matchesQuery("Compare with server", "SERVER")).toBe(true);
    expect(matchesQuery("Compare with server", "missing")).toBe(false);
  });

  it("matches accented text typed without accents, and vice versa", () => {
    expect(matchesQuery("Estimativa de população", "populacao")).toBe(true);
    expect(matchesQuery("Estimativa de populacao", "população")).toBe(true);
  });

  it("requires every token but ignores their order", () => {
    expect(matchesQuery("validations features june 2026", "june validations")).toBe(true);
    expect(matchesQuery("validations features june 2026", "june 2027")).toBe(false);
  });
});
