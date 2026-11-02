/**
 * How one solution of a seed study's panel is drawn.
 *
 * A solution means something different in each problem: relay positions in
 * P1, a choice among candidate positions in P2 and P3, a route of a mobile
 * sink in P4, a decision vector in P0. `solutionScene` reads the instance and
 * one chromosome into what to draw, one reader per problem; `topologyOption`
 * turns a scene into an ECharts option. Both are pure, so each reading can be
 * tested without a canvas.
 */
import type { EChartsOption } from "echarts";
import type { chartPalette } from "./chartTheme";

type Palette = ReturnType<typeof chartPalette>;
type Json = Record<string, unknown>;
export type Point = [number, number];
export type Region = [number, number, number, number];

export interface Fact {
  label: string;
  tone?: "ok" | "warn";
  hint?: string;
}

export interface NetworkScene {
  kind: "network";
  region: Region;
  /** The fixed sink, or the base of P4's mobile sink. */
  base: { at: Point; label: string } | null;
  /** Fixed motes the solution deploys. */
  motes: Point[];
  motesLabel: string;
  /** Fixed motes the instance gives (P4), which the solution does not move. */
  given: Point[];
  /** Positions the solution could have used and did not. */
  unused: Point[];
  unusedLabel: string;
  links: [Point, Point][];
  linksLabel: string;
  /** What a link means here, said under the chart. */
  linksNote: string;
  /**
   * A radius drawn around every deployed mote, and around the sink when the
   * sink takes part (reach does, sensing does not). Off until asked for.
   */
  disks: { radius: number; label: string; aroundBase: boolean } | null;
  paths: { name: string; points: Point[] }[];
  targets: { at: Point; covered: number }[];
  required: number;
  /** P4: the stops in visiting order, base included. */
  route: { at: Point; dwell: number; location: number }[];
  facts: Fact[];
}

export interface VectorScene {
  kind: "vector";
  values: number[];
  facts: Fact[];
}

export interface EmptyScene {
  kind: "none";
  message: string;
  facts: Fact[];
}

export type SolutionScene = NetworkScene | VectorScene | EmptyScene;

// -------------------------------------------------------
// Reading the instance
// -------------------------------------------------------

function point(value: unknown): Point | null {
  if (Array.isArray(value) && value.length >= 2) {
    const [x, y] = value.map(Number);
    if (Number.isFinite(x) && Number.isFinite(y)) return [x as number, y as number];
  }
  if (value && typeof value === "object") {
    const { x, y } = value as { x?: unknown; y?: unknown };
    if (Number.isFinite(Number(x)) && Number.isFinite(Number(y))) return [Number(x), Number(y)];
  }
  return null;
}

function points(value: unknown): Point[] {
  return Array.isArray(value) ? value.map(point).filter((p): p is Point => p !== null) : [];
}

function number(value: unknown): number | null {
  const n = Number(value);
  return value !== null && value !== undefined && Number.isFinite(n) ? n : null;
}

function distance(a: Point, b: Point): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function macLabel(chromosome: Json): Fact | null {
  const mac = chromosome.mac_protocol;
  if (mac === undefined || mac === null) return null;
  return { label: `MAC ${mac === 1 ? "tsch" : "csma"}` };
}

/** Evaluate a parametric coordinate expression (numpy-style) at time t. */
function evalExpr(expr: string, t: number): number {
  const js = expr
    .replace(/np\.cos/g, "Math.cos")
    .replace(/np\.sin/g, "Math.sin")
    .replace(/np\.tan/g, "Math.tan")
    .replace(/np\.pi/g, "Math.PI")
    .replace(/np\.sqrt/g, "Math.sqrt")
    .replace(/np\.exp/g, "Math.exp")
    .replace(/np\.log/g, "Math.log")
    .replace(/np\.abs/g, "Math.abs");
  return new Function("t", `return ${js}`)(t) as number;
}

/** A mobile node's trajectory, sampled along each segment. */
export function samplePath(segments: [string, string][], closed: boolean, steps = 60): Point[] {
  const out: Point[] = [];
  for (const [xExpr, yExpr] of segments) {
    const segment: Point[] = [];
    for (let i = 0; i <= steps; i++) {
      try {
        const p: Point = [evalExpr(xExpr, i / steps), evalExpr(yExpr, i / steps)];
        if (Number.isFinite(p[0]) && Number.isFinite(p[1])) segment.push(p);
      } catch {
        // an expression this page cannot evaluate is left out
      }
    }
    if (out.length > 0) segment.shift(); // the junction is already there
    out.push(...segment);
  }
  const first = out[0];
  if (closed && first) out.push([first[0], first[1]]);
  return out;
}

function mobilePaths(problem: Json): NetworkScene["paths"] {
  const nodes = Array.isArray(problem.mobile_nodes) ? (problem.mobile_nodes as Json[]) : [];
  return nodes
    .map((node, i) => ({
      name: String(node.name ?? `mobile ${i + 1}`),
      points: samplePath(
        Array.isArray(node.path_segments) ? (node.path_segments as [string, string][]) : [],
        Boolean(node.is_closed),
      ),
    }))
    .filter((path) => path.points.length > 1);
}

/** Every pair within reach, and how many motes reach the sink in hops of that length. */
function reachGraph(base: Point | null, motes: Point[], reach: number | null) {
  const nodes = base ? [base, ...motes] : motes;
  const links: [Point, Point][] = [];
  const neighbours = nodes.map(() => [] as number[]);
  if (reach !== null) {
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        if (distance(nodes[i] as Point, nodes[j] as Point) <= reach) {
          links.push([nodes[i] as Point, nodes[j] as Point]);
          neighbours[i]?.push(j);
          neighbours[j]?.push(i);
        }
      }
    }
  }
  let joined: number | null = null;
  if (base && reach !== null) {
    const seen = new Set([0]);
    const queue = [0];
    while (queue.length) {
      for (const next of neighbours[queue.shift() as number] ?? []) {
        if (!seen.has(next)) {
          seen.add(next);
          queue.push(next);
        }
      }
    }
    joined = seen.size - 1;
  }
  return { links, joined };
}

function boundsOf(problem: Json, all: Point[]): Region {
  const region = problem.region;
  if (Array.isArray(region) && region.length === 4 && region.every((v) => Number.isFinite(Number(v)))) {
    return region.map(Number) as Region;
  }
  if (all.length === 0) return [-100, -100, 100, 100];
  const xs = all.map((p) => p[0]);
  const ys = all.map((p) => p[1]);
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}

function emptyNetwork(problem: Json): Omit<NetworkScene, "region" | "facts"> {
  const sink = point(problem.sink);
  return {
    kind: "network",
    base: sink ? { at: sink, label: "Sink" } : null,
    motes: [],
    motesLabel: "Relays",
    given: [],
    unused: [],
    unusedLabel: "Unused candidates",
    links: [],
    linksLabel: "In reach",
    linksNote:
      "Links join motes within the radius of reach: they are possible links, not the routes the protocol chose.",
    disks: null,
    paths: mobilePaths(problem),
    targets: [],
    required: 0,
    route: [],
  };
}

function connectivityFact(joined: number | null, total: number): Fact | null {
  if (joined === null) return null;
  return {
    label: `${joined}/${total} reach the sink`,
    tone: joined === total ? "ok" : "warn",
    hint: "Motes with a multi-hop path to the sink, every hop within the radius of reach.",
  };
}

function withRegion(problem: Json, scene: Omit<NetworkScene, "region" | "facts">, facts: (Fact | null)[]): NetworkScene {
  const all = [
    ...(scene.base ? [scene.base.at] : []),
    ...scene.motes,
    ...scene.given,
    ...scene.unused,
    ...scene.targets.map((t) => t.at),
    ...scene.paths.flatMap((p) => p.points),
  ];
  return { ...scene, region: boundsOf(problem, all), facts: facts.filter((f): f is Fact => f !== null) };
}

// -------------------------------------------------------
// One reader per problem
// -------------------------------------------------------

function vectorScene(_problem: Json, chromosome: Json): SolutionScene {
  const values = Array.isArray(chromosome.x) ? chromosome.x.map(Number) : [];
  return {
    kind: "vector",
    values,
    facts: [{ label: `${values.length} decision variables`, hint: "An analytical benchmark: no network to draw." }],
  };
}

/** P1: relays placed freely in the region. */
function relayScene(problem: Json, chromosome: Json): SolutionScene {
  const scene = emptyNetwork(problem);
  const reach = number(problem.radius_of_reach);
  scene.motes = points(chromosome.relays);
  const { links, joined } = reachGraph(scene.base?.at ?? null, scene.motes, reach);
  scene.links = links;
  if (reach !== null) scene.disks = { radius: reach, label: "Radius of reach", aroundBase: true };
  return withRegion(problem, scene, [
    { label: `${scene.motes.length} relays` },
    macLabel(chromosome),
    connectivityFact(joined, scene.motes.length),
    scene.paths.length ? { label: `${scene.paths.length} mobile nodes` } : null,
  ]);
}

/** P2 and P3: a selection over the candidate positions. */
function maskScene(problem: Json, chromosome: Json): SolutionScene {
  const scene = emptyNetwork(problem);
  const candidates = points(problem.candidates);
  const mask = Array.isArray(chromosome.mask) ? chromosome.mask.map(Number) : [];
  scene.motes = candidates.filter((_, i) => mask[i] === 1);
  scene.unused = candidates.filter((_, i) => mask[i] !== 1);
  scene.motesLabel = "Selected candidates";
  const reach = number(problem.radius_of_reach);
  const { links, joined } = reachGraph(scene.base?.at ?? null, scene.motes, reach);
  scene.links = links;

  const facts: (Fact | null)[] = [
    { label: `${scene.motes.length} of ${candidates.length} candidates` },
    macLabel(chromosome),
    connectivityFact(joined, scene.motes.length),
  ];

  const cover = number(problem.radius_of_cover);
  const targets = points(problem.targets);
  if (targets.length > 0 && cover !== null) {
    // P3: each target must be within the sensing radius of k deployed motes.
    scene.required = number(problem.k_required) ?? 1;
    scene.targets = targets.map((at) => ({
      at,
      covered: scene.motes.filter((m) => distance(m, at) <= cover).length,
    }));
    scene.disks = { radius: cover, label: "Radius of coverage", aroundBase: false };
    const met = scene.targets.filter((t) => t.covered >= scene.required).length;
    facts.push({
      label: `${met}/${targets.length} targets covered ${scene.required}×`,
      tone: met === targets.length ? "ok" : "warn",
      hint: "Targets within the radius of coverage of at least k selected motes.",
    });
  } else if (reach !== null) {
    scene.disks = { radius: reach, label: "Radius of reach", aroundBase: true };
  }
  if (scene.paths.length) facts.push({ label: `${scene.paths.length} mobile nodes` });
  return withRegion(problem, scene, facts);
}

/** P4: the route of a mobile sink over stopping locations, and how long it stays. */
function routeScene(problem: Json, chromosome: Json): SolutionScene {
  const scene = emptyNetwork(problem);
  const base = point(problem.sink_base);
  scene.base = base ? { at: base, label: "Base" } : null;
  scene.given = points(problem.nodes);
  scene.motesLabel = "Stops";
  scene.unusedLabel = "Unvisited locations";
  scene.linksLabel = "Collected at a stop";
  scene.linksNote =
    "Dotted lines join each visited stop to the motes the instance lists as visible from it. Larger stops last longer.";

  const locations = Array.isArray(problem.sojourns) ? (problem.sojourns as Json[]) : [];
  const route = Array.isArray(chromosome.route) ? chromosome.route.map(Number) : [];
  const dwell = Array.isArray(chromosome.sojourn_times) ? chromosome.sojourn_times.map(Number) : [];
  const visited = new Set(route);

  scene.route = route
    .map((location, i) => {
      const at = point(locations[location]?.position);
      return at ? { at, dwell: dwell[i] ?? 0, location } : null;
    })
    .filter((stop): stop is NetworkScene["route"][number] => stop !== null);
  scene.motes = [...visited]
    .map((location) => point(locations[location]?.position))
    .filter((p): p is Point => p !== null);
  scene.unused = locations
    .filter((_, i) => !visited.has(i))
    .map((location) => point(location.position))
    .filter((p): p is Point => p !== null);

  const reached = new Set<number>();
  for (const location of visited) {
    const at = point(locations[location]?.position);
    const visible = locations[location]?.visibleNodes;
    if (!at || !Array.isArray(visible)) continue;
    for (const index of visible.map(Number)) {
      const mote = scene.given[index];
      if (!mote) continue;
      reached.add(index);
      scene.links.push([at, mote]);
    }
  }

  const total = dwell.reduce((sum, t) => sum + (Number.isFinite(t) ? t : 0), 0);
  return withRegion(problem, scene, [
    { label: `${visited.size} of ${locations.length} stops visited` },
    { label: `${Math.round(total)} s of stops`, hint: "Sum of the sojourn times along the route." },
    macLabel(chromosome),
    {
      label: `${reached.size}/${scene.given.length} motes in reach of a stop`,
      tone: reached.size === scene.given.length ? "ok" : "warn",
      hint: "Motes visible from at least one visited stop.",
    },
  ]);
}

const READERS: Record<string, (problem: Json, chromosome: Json) => SolutionScene> = {
  problem0: vectorScene,
  problem1: relayScene,
  problem2: maskScene,
  problem3: maskScene,
  problem4: routeScene,
};

export function solutionScene(problem: Json, chromosome: Json): SolutionScene {
  const name = String(problem.name ?? "");
  const read = READERS[name];
  if (!read) {
    return { kind: "none", message: `There is no drawing for ${name || "this problem"} yet.`, facts: [] };
  }
  return read(problem, chromosome);
}

// -------------------------------------------------------
// Chart
// -------------------------------------------------------

const GRID = { left: 56, right: 24, top: 20, bottom: 56 };
const MOBILE_COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4"];

/** The region framed at the same scale on both axes, so distances read true. */
export function equalScale(region: Region, width: number, height: number) {
  const [xmin, ymin, xmax, ymax] = region;
  const plotW = Math.max(1, width - GRID.left - GRID.right);
  const plotH = Math.max(1, height - GRID.top - GRID.bottom);
  const scale = Math.max((xmax - xmin) / plotW, (ymax - ymin) / plotH) * 1.08;
  const cx = (xmin + xmax) / 2;
  const cy = (ymin + ymax) / 2;
  return {
    xMin: cx - (scale * plotW) / 2,
    xMax: cx + (scale * plotW) / 2,
    yMin: cy - (scale * plotH) / 2,
    yMax: cy + (scale * plotH) / 2,
  };
}

function circle(center: Point, radius: number, steps = 48): Point[] {
  return Array.from({ length: steps + 1 }, (_, i) => {
    const a = (2 * Math.PI * i) / steps;
    return [center[0] + radius * Math.cos(a), center[1] + radius * Math.sin(a)] as Point;
  });
}

function xy(p: Point): string {
  return `(${p[0].toFixed(1)}, ${p[1].toFixed(1)})`;
}

/** A line drawn as several series sharing a name, so the legend toggles them together. */
function lineSeries(name: string, data: Point[], style: Record<string, unknown>, z: number) {
  return {
    name,
    type: "line" as const,
    data,
    symbol: "none",
    silent: true,
    z,
    lineStyle: style,
    // The legend draws its icon in the item colour, not the line's.
    itemStyle: { color: style.color },
    tooltip: { show: false },
    legendHoverLink: false,
  };
}

function networkOption(scene: NetworkScene, c: Palette, width: number, height: number): EChartsOption {
  const { xMin, xMax, yMin, yMax } = equalScale(scene.region, width, height);
  const [x0, y0, x1, y1] = scene.region;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const series: any[] = [
    {
      ...lineSeries("Region", [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]],
        { color: c.muted, width: 1.25, type: "dashed" }, 0),
    },
  ];
  const legend: string[] = [];
  const hidden: Record<string, boolean> = {};

  if (scene.disks) {
    const centres = [...(scene.base && scene.disks.aroundBase ? [scene.base.at] : []), ...scene.motes];
    for (const centre of centres) {
      series.push(lineSeries(scene.disks.label, circle(centre, scene.disks.radius),
        { color: c.hv, width: 1, opacity: 0.35 }, 1));
    }
    legend.push(scene.disks.label);
    hidden[scene.disks.label] = false;
  }

  for (const [a, b] of scene.links) {
    series.push(lineSeries(scene.linksLabel, [a, b],
      { color: c.hv, width: 1.25, opacity: 0.45, type: scene.route.length ? "dotted" : "solid" }, 2));
  }
  if (scene.links.length) legend.push(scene.linksLabel);

  scene.paths.forEach((path, i) => {
    const color = MOBILE_COLORS[i % MOBILE_COLORS.length];
    series.push({ ...lineSeries(path.name, path.points, { color, width: 2, type: "dashed" }, 3), itemStyle: { color } });
    legend.push(path.name);
  });

  if (scene.route.length > 1) {
    const arrows = scene.route.slice(1).map((stop, i) => {
      const from = (scene.route[i] as NetworkScene["route"][number]).at;
      return {
        value: [(from[0] + stop.at[0]) / 2, (from[1] + stop.at[1]) / 2],
        // "arrow" points up and turns counter-clockwise, so a heading of θ is θ − 90°.
        symbolRotate: (Math.atan2(stop.at[1] - from[1], stop.at[0] - from[0]) * 180) / Math.PI - 90,
      };
    });
    series.push({ ...lineSeries("Route", scene.route.map((s) => s.at), { color: c.igd, width: 2 }, 4), itemStyle: { color: c.igd } });
    series.push({
      name: "Route", type: "scatter", data: arrows, symbol: "arrow", symbolSize: 11,
      itemStyle: { color: c.igd }, silent: true, z: 5, tooltip: { show: false },
    });
    legend.push("Route");
  }

  if (scene.unused.length) {
    series.push({
      name: scene.unusedLabel,
      type: "scatter",
      data: scene.unused.map((p) => ({ value: p, tip: `${scene.unusedLabel.toLowerCase()} ${xy(p)}` })),
      symbolSize: 7,
      itemStyle: { color: "transparent", borderColor: c.muted, borderWidth: 1.25 },
      z: 6,
    });
    legend.push(scene.unusedLabel);
  }

  if (scene.given.length) {
    series.push({
      name: "Motes",
      type: "scatter",
      data: scene.given.map((p, i) => ({ value: p, tip: `mote ${i} ${xy(p)}` })),
      symbol: "rect",
      symbolSize: 8,
      itemStyle: { color: c.muted },
      z: 6,
    });
    legend.push("Motes");
  }

  if (scene.targets.length) {
    series.push({
      name: "Targets",
      type: "scatter",
      data: scene.targets.map((t, i) => ({
        value: t.at,
        tip: `target T${i + 1} ${xy(t.at)}<br>covered by ${t.covered} (needs ${scene.required})`,
        itemStyle: { color: t.covered >= scene.required ? "#10b981" : "#ef4444" },
      })),
      symbol: "pin",
      symbolSize: 16,
      itemStyle: { color: "#10b981" },
      z: 7,
    });
    legend.push("Targets");
  }

  if (scene.motes.length) {
    const stops = new Map(scene.route.map((s, order) => [s.location, { ...s, order }]));
    const dwellMax = Math.max(1, ...scene.route.map((s) => s.dwell));
    series.push({
      name: scene.motesLabel,
      type: "scatter",
      data: scene.route.length
        ? [...stops.values()].map((s) => ({
            value: s.at,
            symbolSize: 8 + 10 * Math.sqrt(s.dwell / dwellMax),
            tip: `stop L${s.location} ${xy(s.at)}<br>stays ${s.dwell.toFixed(1)} s`,
          }))
        : scene.motes.map((p, i) => ({ value: p, tip: `${scene.motesLabel.toLowerCase()} #${i + 1} ${xy(p)}` })),
      symbolSize: 11,
      itemStyle: { color: scene.route.length ? c.igd : c.hv, borderColor: c.tooltip, borderWidth: 1.5 },
      z: 8,
    });
    legend.push(scene.motesLabel);
  }

  if (scene.base) {
    series.push({
      name: scene.base.label,
      type: "scatter",
      data: [{
        value: scene.base.at,
        tip: `${scene.base.label.toLowerCase()} ${xy(scene.base.at)}`,
        label: { show: true, formatter: scene.base.label, position: "top", fontSize: 11, fontWeight: "bold", color: c.gd },
      }],
      symbol: "diamond",
      symbolSize: 18,
      itemStyle: { color: c.gd, borderColor: c.tooltip, borderWidth: 1.5 },
      z: 9,
    });
    legend.push(scene.base.label);
  }

  const axis = (min: number, max: number) => ({
    type: "value" as const,
    min,
    max,
    // The ends are wherever the equal scale put them, not round numbers.
    axisLabel: {
      color: c.muted,
      fontSize: 11,
      showMinLabel: false,
      showMaxLabel: false,
      formatter: (v: number) => String(Math.round(v)),
    },
    axisLine: { lineStyle: { color: c.grid } },
    splitLine: { lineStyle: { color: c.grid } },
  });

  return {
    backgroundColor: c.bg,
    animation: false,
    grid: GRID,
    tooltip: {
      trigger: "item",
      backgroundColor: c.tooltip,
      borderColor: c.tooltipBorder,
      textStyle: { color: c.text, fontSize: 12 },
      formatter: (p: unknown) => String((p as { data?: { tip?: string } }).data?.tip ?? ""),
    },
    legend: {
      bottom: 4,
      type: "scroll",
      data: legend,
      selected: hidden,
      textStyle: { color: c.text, fontSize: 11 },
    },
    xAxis: axis(xMin, xMax),
    yAxis: axis(yMin, yMax),
    series,
  };
}

function vectorOption(scene: VectorScene, c: Palette): EChartsOption {
  return {
    backgroundColor: c.bg,
    animation: false,
    grid: { left: 48, right: 24, top: 20, bottom: 40 },
    tooltip: {
      trigger: "item",
      backgroundColor: c.tooltip,
      borderColor: c.tooltipBorder,
      textStyle: { color: c.text, fontSize: 12 },
    },
    xAxis: {
      type: "category",
      data: scene.values.map((_, i) => `x${i}`),
      axisLabel: { color: c.muted, fontSize: 11 },
      axisLine: { lineStyle: { color: c.grid } },
    },
    yAxis: {
      type: "value",
      min: 0,
      max: 1,
      axisLabel: { color: c.muted, fontSize: 11 },
      splitLine: { lineStyle: { color: c.grid } },
    },
    series: [{ name: "x", type: "bar", data: scene.values, itemStyle: { color: c.hv }, barMaxWidth: 28 }],
  };
}

export function topologyOption(scene: SolutionScene, c: Palette, width: number, height: number): EChartsOption {
  if (scene.kind === "network") return networkOption(scene, c, width, height);
  if (scene.kind === "vector") return vectorOption(scene, c);
  return {};
}
