/**
 * Reading the pool as a graph: master-node → hosts → workers.
 *
 * Pure, so the three things the page has to get right are testable without a
 * canvas: what colour a status is, where each node goes, and what a tooltip
 * says — the last one includes names and messages typed by operators or
 * returned by remote machines, so it is also where HTML gets escaped.
 */
import type {
  MasterLiveness,
  TopologyDto,
  TopologyHost,
  TopologyRunningExperiment,
  TopologyWorker,
} from "../types/infra";

/** Five readings, shared by nodes, edges and the legend. */
export type Tone = "ok" | "busy" | "warn" | "bad" | "off";

export const TONE_LABEL: Record<Tone, string> = {
  ok: "available / reachable",
  busy: "running a simulation",
  warn: "degraded, held or being removed",
  bad: "unreachable or failing",
  off: "disabled",
};

/** The colour of each reading, per theme — shared by the graph and its legend. */
export function toneColors(dark: boolean): Record<Tone, string> {
  return {
    ok: dark ? "#a6e3a1" : "#16a34a",
    busy: dark ? "#89b4fa" : "#2563eb",
    warn: dark ? "#f9e2af" : "#d97706",
    bad: dark ? "#f38ba8" : "#dc2626",
    off: dark ? "#6c7086" : "#94a3b8",
  };
}

export function masterTone(status: MasterLiveness): Tone {
  if (status === "alive") return "ok";
  if (status === "stale") return "bad";
  return "off";
}

export function hostTone(host: Pick<TopologyHost, "status" | "enabled" | "withdrawing">): Tone {
  if (!host.enabled || host.status === "disabled") return "off";
  if (host.withdrawing) return "warn";
  if (host.status === "online") return "ok";
  if (host.status === "degraded") return "warn";
  return "bad";
}

/**
 * A worker's reading.
 *
 * Occupied comes first: a worker finishing a simulation after being switched
 * off is still running it, and saying "disabled" would hide the run. Held
 * comes next — it is out of the pool *and* holding something unresolved,
 * which is a different thing to act on than an ordinary switch-off.
 */
export function workerTone(worker: Pick<TopologyWorker, "status" | "enabled" | "held">): Tone {
  if (worker.status === "busy" || worker.status === "reserved") return "busy";
  if (worker.held) return "warn";
  if (!worker.enabled || worker.status === "disabled") return "off";
  if (worker.status === "available") return "ok";
  return "bad";
}

export function workerLabel(worker: Pick<TopologyWorker, "status" | "enabled" | "held">): string {
  if (worker.status === "busy") return "running";
  if (worker.status === "reserved") return "reserved";
  if (worker.held) return "held";
  if (!worker.enabled || worker.status === "disabled") return "disabled";
  return worker.status || "unknown";
}

/**
 * Colours for telling concurrent experiments apart. Mid-tones that read on
 * both themes, and none of them the "running" blue or the "failing" red, so
 * an experiment's colour is never mistaken for a status.
 */
export const EXPERIMENT_COLORS = [
  "#f97316",
  "#8b5cf6",
  "#14b8a6",
  "#ec4899",
  "#84cc16",
  "#eab308",
  "#06b6d4",
  "#a16207",
];

/**
 * A stable colour per running experiment.
 *
 * Assigned in id order: ids grow with creation time, so an experiment that
 * starts later takes the next colour and the ones already on screen keep
 * theirs. Ordering by the list the API returns — busiest first — would
 * repaint every experiment whenever their worker counts crossed.
 */
export function experimentColors(running: TopologyRunningExperiment[]): Record<string, string> {
  const ids = [...running.map((e) => e.id)].sort();
  return Object.fromEntries(
    ids.map((id, i) => [id, EXPERIMENT_COLORS[i % EXPERIMENT_COLORS.length] as string]),
  );
}

export interface NodeLayout {
  id: string;
  kind: "master" | "host" | "worker";
  x: number;
  y: number;
  hostId?: string;
}

export interface EdgeLayout {
  source: string;
  target: string;
}

/** Workers per line beside a host before it wraps onto a second line. */
export const WORKERS_PER_ROW = 10;

/** Columns of the drawing, in layout units. */
export const LAYOUT_X = { master: 0, host: 2.4, firstWorker: 4.3 };

export interface TopologyLayout {
  nodes: NodeLayout[];
  edges: EdgeLayout[];
  /** Lines the drawing takes — the canvas height follows it. */
  rows: number;
  /** Data range to show, padded so the boxes at the edges are not cut. */
  bounds: { xMin: number; xMax: number; yMin: number; yMax: number };
}

/**
 * Where every node goes: master on the left, one line per host, and each
 * host's workers in a row beside it.
 *
 * Left to right because the pool grows by adding hosts: a row of hosts
 * across the top ran out of width at ten, with every box overlapping its
 * neighbours, while a column of hosts only grows the drawing downward. A host
 * with more than {@link WORKERS_PER_ROW} workers takes a second line.
 *
 * Deterministic, so a refresh redraws the same picture with new colours
 * instead of shuffling it — a force layout would move the nodes every few
 * seconds while a run is on.
 */
export function layoutTopology(topology: Pick<TopologyDto, "hosts">): TopologyLayout {
  const nodes: NodeLayout[] = [];
  const edges: EdgeLayout[] = [];
  let row = 0;
  let widest = 0;
  for (const host of topology.hosts) {
    const lines = Math.max(1, Math.ceil(host.workers.length / WORKERS_PER_ROW));
    nodes.push({
      id: hostNodeId(host.id),
      kind: "host",
      x: LAYOUT_X.host,
      y: row + (lines - 1) / 2,
    });
    edges.push({ source: MASTER_NODE_ID, target: hostNodeId(host.id) });
    host.workers.forEach((worker, i) => {
      const column = i % WORKERS_PER_ROW;
      widest = Math.max(widest, column + 1);
      nodes.push({
        id: workerNodeId(worker.id),
        kind: "worker",
        x: LAYOUT_X.firstWorker + column,
        y: row + Math.floor(i / WORKERS_PER_ROW),
        hostId: host.id,
      });
      edges.push({ source: hostNodeId(host.id), target: workerNodeId(worker.id) });
    });
    row += lines;
  }
  const rows = Math.max(1, row);
  nodes.unshift({ id: MASTER_NODE_ID, kind: "master", x: LAYOUT_X.master, y: (rows - 1) / 2 });
  return {
    nodes,
    edges,
    rows,
    bounds: {
      xMin: LAYOUT_X.master - 0.9,
      xMax: LAYOUT_X.firstWorker + Math.max(1, widest) - 0.4,
      yMin: -0.6,
      yMax: rows - 0.4,
    },
  };
}

export const MASTER_NODE_ID = "master";
export const hostNodeId = (id: string) => `host:${id}`;
export const workerNodeId = (id: string) => `worker:${id}`;

/** Counts under a host's label: how much of it is working right now. */
export function hostLoad(host: Pick<TopologyHost, "workers">): {
  running: number;
  total: number;
} {
  return {
    running: host.workers.filter((w) => w.status === "busy" || w.status === "reserved").length,
    total: host.workers.length,
  };
}

/**
 * How long a run has been on its worker, against the server's clock.
 *
 * The response carries its own `generated_at`; measuring against the
 * browser's clock would show a run as starting in the future on a laptop a
 * minute behind.
 */
export function elapsedSeconds(since: string | null | undefined, now: string): number | null {
  if (!since) return null;
  const start = Date.parse(since);
  const end = Date.parse(now);
  if (Number.isNaN(start) || Number.isNaN(end)) return null;
  return Math.max(0, Math.round((end - start) / 1000));
}

export function formatElapsed(seconds: number | null): string {
  if (seconds === null) return "—";
  if (seconds < 60) return `${seconds} s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes} min ${seconds % 60} s`;
  return `${Math.floor(minutes / 60)} h ${minutes % 60} min`;
}

/** Names and messages come from operators and remote machines. */
export function escapeHtml(text: string | null | undefined): string {
  return String(text ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

export function masterTooltip(master: TopologyDto["master"]): string {
  const heading = "<b>master-node</b>";
  if (master.status === "unknown") {
    return `${heading}<br/>No enabled host to observe it by.`;
  }
  const age = formatElapsed(master.age_sec === undefined ? null : Math.round(master.age_sec));
  return master.status === "alive"
    ? `${heading} — running<br/>last health check ${age} ago`
    : `${heading} — <b>not checking hosts</b><br/>last health check ${age} ago ` +
        `(expected every few seconds; stale after ${master.stale_after_sec} s)`;
}

export function hostTooltip(host: TopologyHost): string {
  const load = hostLoad(host);
  const lines = [
    `<b>${escapeHtml(host.name)}</b> — ${escapeHtml(host.status)}` +
      (host.withdrawing ? " · being removed" : ""),
    `${load.running} of ${load.total} worker(s) running`,
  ];
  if (!host.schedulable && host.enabled) lines.push("not taking new work");
  if (host.message) lines.push(`<i>${escapeHtml(host.message)}</i>`);
  return lines.join("<br/>");
}

export function workerTooltip(worker: TopologyWorker, now: string): string {
  const lines = [`<b>${escapeHtml(worker.name)}</b> — ${workerLabel(worker)}`];
  if (worker.held && worker.held_kind) lines.push(`held: ${escapeHtml(worker.held_kind)}`);
  const run = worker.run;
  if (run) {
    lines.push(`experiment: ${escapeHtml(run.experiment_name ?? run.experiment_id ?? "?")}`);
    if (run.generation_index !== null && run.generation_index !== undefined) {
      lines.push(
        `generation ${run.generation_index}` +
          (run.random_seed !== null && run.random_seed !== undefined
            ? ` · seed ${run.random_seed}`
            : ""),
      );
    }
    lines.push(
      `on this worker for ${formatElapsed(elapsedSeconds(run.assigned_at ?? run.started_at, now))}`,
    );
  } else if (worker.message) {
    lines.push(`<i>${escapeHtml(worker.message)}</i>`);
  }
  return lines.join("<br/>");
}

/** The numbers under the page title. */
export function summarize(topology: Pick<TopologyDto, "hosts">): {
  hosts: number;
  hostsReachable: number;
  workers: number;
  running: number;
  free: number;
} {
  const workers = topology.hosts.flatMap((h) => h.workers);
  return {
    hosts: topology.hosts.length,
    hostsReachable: topology.hosts.filter((h) => hostTone(h) === "ok" || hostTone(h) === "warn")
      .length,
    workers: workers.length,
    running: workers.filter((w) => workerTone(w) === "busy").length,
    free: workers.filter((w) => workerTone(w) === "ok").length,
  };
}
