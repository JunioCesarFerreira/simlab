# SimLab GUI

Front-end for the SimLab experiment platform — a web interface for designing
Wireless Sensor Network (WSN) topologies, launching multi-objective evolutionary
algorithm (MOEA) experiments, analysing the resulting Pareto fronts, and
managing the infrastructure that runs the simulations.

---

## Tech stack

| Layer | Library |
|---|---|
| Framework | Vue 3 (Composition API, `<script setup>`) |
| Language | TypeScript |
| Build | Vite |
| State | Pinia |
| Routing | Vue Router |
| Charts | Apache ECharts 6 + echarts-gl (WebGL 3-D) |
| HTTP | Axios |
| Credential sealing | libsodium-wrappers |
| Tests | Vitest |

> **echarts-gl and ECharts 6.** The locked `echarts-gl@2.1.0` declares
> `echarts@^5.1.2 || ^6.0.0` as its peer dependency. If a 3-D chart breaks after
> an `echarts` upgrade, check that pair first; the 2-D charts do not use it.

---

## Getting started

### Prerequisites

- Node.js 20.19+ or 22.12+ (Vite 7's requirement; 18 is no longer enough, and
  the build fails on it rather than warning). The image builds with Node 22.
- SimLab stack running behind the reverse proxy (default `https://localhost`)

### Environment

The REST API no longer publishes a host port — it is reachable only through
the nginx reverse proxy. Point the dev server at the proxy, in
`gui/simlab/.env`:

```
VITE_API_BASE_URL=https://localhost/api/v1
VITE_API_KEY=simlab-api-key42
```

The proxy's certificate is self-signed, so open `https://localhost/` once and
accept the warning; the browser then trusts that origin for the XHR calls
`npm run dev` makes from `http://localhost:5173`. Those are cross-origin
requests, which work because the API still sends
`Access-Control-Allow-Origin: *`.

In production the SPA is served *by* the proxy, so it uses the relative base
`/api/v1` (set as a build arg in `docker-compose.yaml`) and no CORS is involved.

### Development server

```bash
npm install
npm run dev          # http://localhost:5173
```

### Production build

```bash
npm run build        # output → dist/
npm run preview      # preview the built output locally
```

### Checks

```bash
npm run type-check
npm run lint
npm test
npm run format       # Prettier; rewrites files
```

---

## Application structure

```
src/
├── api/                 # Axios wrappers (experiments, problems, campaigns, …)
├── app/
│   ├── router/          # Vue Router routes
│   └── stores/          # Pinia stores (experiments, detail, campaigns, problem, …)
├── components/
│   ├── campaigns/       # Campaign cards
│   ├── charts/          # The experiment and comparison charts
│   ├── common/          # Dialogs, badges, resizable chart card
│   ├── detail/          # Individual panel, generation rows, DODAG, firmware,
│   │                    # runtime metrics, work distribution
│   ├── experiments/     # Experiment cards and filters
│   ├── infra/           # Host cards, worker table, occupancy, machines, topology, forms
│   ├── layout/          # Sidebar navigation, theme toggle
│   ├── problem-editor/  # Canvas editor + 5-step launch wizard
│   ├── scalability/     # USL benchmark form, result and charts
│   ├── sources/         # Source file viewer
│   └── synthetic-editor/ # Benchmark editor + 4-step launch wizard
├── composables/         # useEChart, useResizable, useExperimentViewState, useTheme, …
├── lib/                 # GA capabilities, NSGA-III helpers, seed aggregators
├── pages/               # One file per top-level route
├── services/            # Import/export helpers, validators, the default data
│                        # conversion, and the pure modules that mirror a
│                        # decision the backend makes: resourceLevels, placement,
│                        # completeness, secretEnvelope, metricKinds
├── types/               # TypeScript types (simlab, problem, infra, scalability)
├── utils/               # Comparison indicators, non-dominated sort, export
└── workers/             # Web Worker that computes Pareto ranks
```

### Navigation

| Sidebar item | Route | Purpose |
|---|---|---|
| Dashboard | `/dashboard` | Live feed of running / queued / finished experiments |
| Campaigns | `/campaigns` | Group related experiments |
| Experiments | `/experiments` | Full list across all campaigns; direct detail access |
| Problems | `/problems` | WSN problem editor (full-screen canvas) |
| Synthetic | `/synthetic` | Analytical benchmark instances (P0), evaluated in-process |
| Sources | `/sources` | Firmware source-code repositories |
| Compare | `/compare` | Side-by-side comparison of two finished experiments |
| Infrastructure | `/infrastructure` | Hosts and their Cooja workers: register a machine, discover or hand-configure its containers, see which level of availability failed, take one in or out of the pool, and limit a physical machine |
| Topology | `/topology` | The pool as a live graph: master-node, hosts, workers and the experiment each one runs |
| Scalability | `/scalability` | Launch a USL benchmark and read its fit, for the pool and per physical machine |

---

## Research workflow — WSN + multi-objective optimisation

The typical flow from problem design to result analysis follows these five stages.

### 1. Design the WSN problem

Go to **Problems** (`/problems`). The canvas editor lets you:

- Set the simulation region (bounding box in metres).
- Place **candidate sensor positions** (nodes the algorithm may or may not activate).
- Place **target points** (coverage requirements).
- Set a **sink node** position.
- Configure connectivity radii: `radiusOfReach` (communication), `radiusOfInter`
  (interference), `radiusOfCover` (sensing).
- Add **mobile nodes** with parametric routes (line, ellipse, or custom expression).
- Choose the **problem variant** (`problem1`–`problem4`) which governs which
  elements are required. Only P1–P3 can run: the export does not write what P4
  needs (see [Known issues](#known-issues)).
- Set the **coverage level α** (P1, P2) and the coverage degree **k** (P3).

**Saving problems:** use the *Problems* panel (toolbar button) to persist the
current design to the server. Saved problems can be loaded or deleted.
The background image (floor plan, map) is stored alongside the draft.

**Importing / exporting:** the toolbar also supports JSON import/export for
version control or sharing configurations between users.

### 2. Create a campaign (optional)

Go to **Campaigns** and create a new campaign, or create one when the launch
finishes. A campaign is a named container
for a set of related experiments — for example, all runs that compare NSGA-II
vs NSGA-III on the same problem.

### 3. Launch an experiment (5-step wizard)

In the problem editor, click **🚀** (enabled once a sink is placed). The wizard
launches the problem that is open in the editor:

| Step | Name | What to configure |
|---|---|---|
| 1 | **Problem** | Review the problem defined in the editor |
| 2 | **Experiment** | Algorithm (NSGA-II / NSGA-III, native or DEAP/pymoo, or random search), one source repository (firmware) per MAC protocol, population size, number of generations, crossover/mutation rates, the operator knobs the problem reads, NSGA-III divisions |
| 3 | **Simulation** | Simulated duration (**minutes**, although the field says seconds), random seeds |
| 4 | **Objectives** | One row per metric to optimise — name of a metric of step 5 (defaults: `latency`, `power_mw`, `throughput`), direction (`Minimize` / `Maximize`), and how its value is combined across seeds (mean / median / trimmed mean / min / max). The row says when min/max judges an individual by its worst or best seed |
| 5 | **Data conversion** | Column names in the Cooja CSV log, plus one [kind](../../docs/markdown/FIRMWARE_METRICS.md#21-what-a-data-conversion-does) per metric, from the list the evaluator knows. The default reads firmware v2 |

When the experiment is created, a dialog offers to add it to a campaign.

#### Tip — reusing seeds across experiments

In Step 3, use **↓ Export** to save the current seed list to a JSON file, then
**↑ Import** on any subsequent experiment to evaluate with the same simulation
seeds — critical for fair algorithm comparisons. The initial population depends
on the *algorithm random seed* of step 2, not on this list.

### 4. Monitor experiments

The **Experiment Detail** page (`/experiments/:id`) updates live while the job
runs. It shows:

- Status badge, system message and progress bar (generations completed / total).
- **Parameters card** with the full algorithm and simulation configuration.
- **Pareto Front** chart — switchable between 2-D scatter (coloured by
  non-domination rank) and 3-D WebGL scatter (when ≥ 3 objectives).
- **HV, GD & IGD per generation** — hypervolume, generational distance and
  inverted generational distance tracked generation by generation.
- **Objectives evolution** — normalised best-per-generation line chart for each
  objective.
- **Parallel coordinates** — each Pareto solution as a poly-line across all
  objective axes (↑ = better for every axis); lines are coloured by the first
  objective's value.
- **Runtime Metrics** — CPU and memory of the run
  ([RUNTIME_METRICS.md](../../docs/markdown/RUNTIME_METRICS.md)).
- **Generations table** — expandable rows with per-individual chromosome,
  objectives, topology image, and simulations list.
- **Firmware** — the copy of the firmware the run used
  ([FIRMWARE_TRACEABILITY.md](../../docs/markdown/FIRMWARE_TRACEABILITY.md)).
- **Work distribution** — how the simulations spread over hosts and containers.

The four chart cards are **drag-resizable** (grab the bottom edge). Their
heights, the 2-D / 3-D toggle, the front's scope, pinned points and axis
selections are all **persisted per experiment** in `sessionStorage`, so they
survive navigation back to the page.

**Clicking a Pareto solution** in any chart (scatter or parallel coordinates)
opens the **Individual Detail** side panel: objective values, topology image,
what the evaluation rests on (replica completeness), chromosome key-value
table, and the list of associated simulations.

### 5. Compare experiments

Go to **Compare** (`/compare`), select a campaign and two finished experiments,
and click **Compare**. The page shows:

- **Parameters table** — side-by-side diff of all algorithm / simulation parameters.
- **Pareto Front comparison** — both fronts overlaid on the same 2-D chart
  (axis-selectable) or 3-D WebGL chart.
- **Convergence charts** — hypervolume and generational distance per generation
  for both experiments on the same axes, with a selector of the measured
  population. Each experiment uses its own references, so in non-synthetic
  runs the two curves are not on a common scale.
- **Quality indicators** table:

  | Indicator | Meaning |
  |---|---|
  | HV, GD | Each experiment's own, from the API, against its own references |
  | C(A→B) | Coverage — fraction of B's solutions dominated by A |
  | C(B→A) | Coverage in the reverse direction |
  | IGD+(A→B), IGD+(B→A) | Inverted Generational Distance Plus of one final front against the other — proximity + spread |
  | ε(A→B), ε(B→A) | Additive epsilon: how much one front needs to shift to weakly dominate every point of the other |
  | Spacing | Schott's spacing of each front — uniformity of distribution (lower = more uniform) |

  All but HV and GD are computed in the browser from the final fronts, in raw
  units: objectives of larger magnitude dominate IGD+, ε and Spacing.

---

## Chart reference

### Pareto Front (2-D)

Scatter chart coloured by Pareto rank (Front 1 = blue, Front 2 = emerald, …).
When generation data is available, all population members are shown in muted
colours behind the front. Axis selectors appear when there are more than 2
objectives. Click any point to open the individual detail panel.

**Pin mode** (📌) — click a point to pin it; the pinned marker persists while
you interact with the chart.

### Pareto Front (3-D)

WebGL scatter (echarts-gl). Controls:
- **Axis selectors** — choose which three objectives to display.
- **Pan / Rotate toggle** — swap left-drag between panning and rotating.
- **Niche Lines** — NSGA-III reference directions (visible for `nsga3` variants).
- **Pin mode** — the pins are shared with the 2-D chart.
- **Dominance mode** — click a point to highlight the region it dominates.

The camera position is preserved across data refreshes (polling) so the user's
viewpoint is not reset while the experiment runs.

### HV, GD & IGD per generation

Three side-by-side line charts, all read from `GET /experiments/{id}/hv-gd` —
none of these numbers are computed in the browser.

One **Measured set** selector drives the three:

- **Survivors** (default): the population environmental selection kept, $P_t$.
  A generation without one (random search, batch, older runs) falls back to its
  offspring, and the caption says so.
- **Offspring**: the individuals evaluated in that generation, $Q_t$.
- **Archive**: the non-dominated set of everything evaluated so far.

| Panel | Reads | Lower/higher is better |
|---|---|---|
| **HV** | Hypervolume of the measured set's front | higher |
| **GD** | Mean distance from the front to the nearest reference point, or to the analytical front for a benchmark — *convergence* | lower |
| **IGD / IGD+** | Mean distance from each reference point to the nearest solution — *convergence **and** spread*. IGD+ is the Pareto-compliant variant (Ishibuchi et al., 2015) and bounds IGD from below | lower |

GD and IGD sit on separate axes on purpose: a population that converged onto a
single corner of the front scores a near-zero GD and a large IGD, which is
exactly the failure mode the pair is there to expose.

GD/IGD/IGD+ distances are normalised by the reference front's ideal-nadir range
(for a benchmark, the theoretical one), so objectives of different magnitudes (a latency in ms next to a throughput in
kbps) weigh comparably. Hypervolume keeps its own reference point in raw units
and is unaffected.

A caption under the charts names the reference front. When it is the run's own
final front — every non-synthetic experiment — GD and IGD are self-referential:
they track progress towards *this run's own* result rather than convergence to
the true optimum, so they are not comparable across runs. The caption says so,
because otherwise the curves read as a quality claim they do not support.

The stored front of NSGA-II/III is that of the selected final survivors
(`_final_pareto_front` in `mo-engine`). With *Survivors*, GD and IGD therefore
reach zero on the last generation **by construction**, which says nothing about
convergence. With *Offspring* or *Archive* they need not. The definitions are in
[SYNTHETIC_MODE.md §7](../../docs/markdown/SYNTHETIC_MODE.md#7-convergence-metrics-against-the-true-front).

### Objectives evolution

Normalised best-per-generation value for each objective, rescaled to [0 %, 100 %]
so objectives with different units can be compared on the same axis. The axis
is oriented so that up is better, and only generations that finished `Done`
appear.

### Parallel coordinates

All Pareto solutions as poly-lines. Every axis corresponds to one objective and
is oriented so that **↑ is the better direction** (min-objectives are inverted).
Lines are coloured by the value of the first objective via a continuous heat-map
scale. Click a line to open the individual detail panel.

---

## Keyboard / UX shortcuts

| Action | How |
|---|---|
| Open individual detail | Click a chart data point (scatter or parallel line) |
| Close individual detail | Click the backdrop or the ✕ button |
| Resize a chart | Drag the bottom edge handle of any chart card |
| Switch 2-D / 3-D | Toggle buttons in the Pareto Front card header |
| Pin a solution | Activate 📌 Pin mode then click a point |
| Rotate / pan 3-D scene | Use the Pan button to switch left-drag behaviour |
| Toggle dark / light mode | Button at the bottom of the sidebar |

---

## Development notes

### Adding a new chart

1. Create `src/components/charts/MyChart.vue`.
2. Use the `useEChart(containerRef)` composable — it handles `echarts.init`,
   `ResizeObserver`, and `onBeforeUnmount` cleanup automatically.
3. Set `flex: 1; min-height: 0` on the root `.chart-wrap` so it fills its
   parent flex container correctly when placed inside a resizable `chart-card`.

### Persisting view state

Use `useExperimentViewState(experimentId)` to add a new persisted field:

```ts
// composables/useExperimentViewState.ts
interface ExperimentViewState {
  // ... add your field here with a default in DEFAULTS
  myNewField: string
}
```

The composable reads from and deep-watches to `sessionStorage` automatically.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `VITE_API_BASE_URL` | `/api/v1` in the container build; for `npm run dev`, what `.env` says (`https://localhost/api/v1`) | REST API base URL |
| `VITE_API_KEY` | `simlab-api-key42` in the container build (`SIMLAB_API_KEY` under Compose); none for `npm run dev` besides `.env` | `X-API-Key` header sent with every request. It has to match the API's `SIMLAB_API_KEY` |

Both are fixed at build time.

---

## Known issues

- **Simulated duration is in minutes, labelled as seconds**: in the wizard
  (step 3), in the comparison table (*Sim. duration (s)*) and in the benchmark
  form (*Simulated duration (s)*). The value is stored and used as minutes.
- **The totals do not count generation 0.** A run has
  `number_of_generations + 1` generations, so *Completed generations* can read
  11 / 10 and *expected* simulations is one population short.
- **P4 cannot be launched**: the export writes neither `nodes`, `sink_base` nor
  `sojourns`. See [the problems](../../mo-engine/lib/problem/README.md#known-limitations).
- **The editor has no field for P3's `g`**, which the engine does not read either.
- **Two hints are inexact:** *pm_tau* is the probability of mutating the whole
  dwell-time vector, not each time, and the coverage hint of α in P2 does not
  say that the sink is left out.
