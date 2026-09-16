import type { HvGdData, PopulationSource } from "../api/metrics";

export function populationLabel(source: PopulationSource | null | undefined): string {
  switch (source) {
    case "survivors": return "Survivors";
    case "offspring": return "Offspring";
    case "archive": return "Archive";
    case "mixed": return "Mixed survivors / offspring";
    default: return "Unknown population";
  }
}

export function populationCaption(data: HvGdData): string {
  if (data.population_source === "mixed") {
    const fallback = data.generations.filter((_, i) => data.population_sources?.[i] === "offspring");
    return `Measured on survivors where available; offspring used for generations ${fallback.join(", ")} because survivor sets are unavailable. This is a mixed series.`;
  }
  if (data.population === "survivors" && data.population_source === "offspring") {
    return "Measured on offspring (Q_t): survivor sets are unavailable for these generations.";
  }
  return `Measured on the ${populationLabel(data.population_source).toLowerCase()}.`;
}

export function metricGenerationLabels(data: HvGdData): string[] {
  return data.generations.map((g, i) => data.population_source === "mixed"
    ? `Gen ${g} (${populationLabel(data.population_sources?.[i]).toLowerCase()})`
    : `Gen ${g}`);
}
