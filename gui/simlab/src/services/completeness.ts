import type { CompletenessDto } from "../types/simlab";

/**
 * Reading an evaluation's completeness.
 *
 * An objective vector says nothing about how much evidence is behind it: a
 * mean over one replica and a mean over five are the same two numbers. The
 * API now publishes the counts, and this turns them into something an
 * operator can act on — which replica is missing, and whether the algorithm
 * was given the vector at all.
 *
 * Pure, and kept out of the component for the same reason the resource levels
 * are: it mirrors a decision the engine makes, and that correspondence is
 * only worth anything if it can be tested against it.
 */

export type CompletenessTone = "ok" | "warn" | "bad" | "unknown";

/** The short answer, in words rather than a state name. */
export function completenessLabel(record?: CompletenessDto | null): string {
  if (!record) return "not recorded";
  const { state, observed, expected } = record;
  const counted =
    typeof observed === "number" && typeof expected === "number"
      ? ` (${observed}/${expected} replicas)`
      : "";
  switch (state) {
    case "complete":
      return `complete${counted}`;
    case "partial":
      return `incomplete${counted}`;
    case "pending":
      return `still running${counted}`;
    case "invalid":
      return `no replica finished${counted}`;
    case "analytical":
      return "evaluated analytically";
    case "infeasible":
      return "rejected by a constraint";
    default:
      // The generation never recorded what it planned, so incompleteness
      // cannot be shown — only what was observed. Saying "complete" here
      // would be a claim about data nobody wrote down.
      return `unknown${counted}`;
  }
}

export function completenessTone(record?: CompletenessDto | null): CompletenessTone {
  if (!record) return "unknown";
  switch (record.state) {
    case "complete":
    case "analytical":
      return "ok";
    case "partial":
    case "pending":
      return "warn";
    case "invalid":
      return "bad";
    case "infeasible":
      // Not a measurement problem: the individual was rejected before any
      // simulation, which is a normal outcome of a constrained search.
      return "unknown";
    default:
      return "unknown";
  }
}

/**
 * Metrics with fewer observations than replicas.
 *
 * A run can finish and not carry every metric, so two objectives of the same
 * individual can rest on different numbers of replicas — which a single
 * "2 of 2" cannot express.
 */
export function thinMetrics(record?: CompletenessDto | null): string[] {
  if (!record?.per_metric || typeof record.expected !== "number") return [];
  return Object.entries(record.per_metric)
    .filter(([, count]) => count !== record.expected)
    .map(([metric, count]) => `${metric}: ${count}/${String(record.expected)}`);
}

/** The lines worth showing under the label, in order of usefulness. */
export function completenessDetail(record?: CompletenessDto | null): string[] {
  if (!record) {
    return ["This evaluation was made before completeness was recorded."];
  }
  const lines: string[] = [];
  if (record.reused) {
    // Worth stating: this individual has no simulations of its own in this
    // generation, and the numbers describe the evaluation they came from.
    lines.push("carried over from an earlier evaluation of the same genome");
  }
  if (record.reason) lines.push(record.reason);
  if (record.missing_seeds?.length) {
    lines.push(`missing seeds: ${record.missing_seeds.join(", ")}`);
  }
  if (record.absent) {
    lines.push(`${record.absent} replica(s) were never created`);
  }
  if (record.attempts && record.expected && record.attempts > record.expected) {
    lines.push(`${record.attempts} runs for ${record.expected} replicas (repeated)`);
  }
  const thin = thinMetrics(record);
  if (thin.length) lines.push(`fewer observations: ${thin.join(", ")}`);
  if (record.objective_used === false) {
    lines.push("the algorithm was given the worst objective for this individual");
  }
  if (record.policy) lines.push(`policy: ${record.policy}`);
  return lines;
}
