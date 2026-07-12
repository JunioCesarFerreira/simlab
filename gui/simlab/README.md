# SimLab GUI

Front-end for the SimLab experiment platform — a web interface for designing
Wireless Sensor Network (WSN) topologies, launching multi-objective evolutionary
algorithm (MOEA) experiments, and analysing the resulting Pareto fronts.

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

> **Known version mismatch — echarts-gl.** `echarts-gl@2.x` officially declares
> `echarts@^5` as its peer dependency, but this project runs `echarts@6`. The
> combination works in practice for the 3-D scatter/surface charts we use, and
> no echarts-gl release supporting ECharts 6 exists yet. If a 3-D chart breaks
> after an `echarts` minor upgrade, suspect this first: either pin `echarts`
> to the last working 6.x version or check whether echarts-gl gained ECharts 6
> support. The 2-D charts are unaffected.

---

## Getting started

### Prerequisites

- Node.js ≥ 18
- SimLab REST API running (default `http://localhost:8000`)

### Environment

Copy `.env` and adjust if needed:

```
VITE_API_BASE_URL=http://localhost:8000/api/v1
VITE_API_KEY=api-password
```

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

### Lint / format

```bash
npm run lint
npm run format
```

---

## Application structure

```
src/
├── api/                 # Axios wrappers (experiments, problems, campaigns, …)
├── app/
│   ├── router/          # Vue Router routes
│   └── stores/          # Pinia stores (experiment detail, problem, repositories)
├── components/
│   ├── charts/          # All ECharts chart components
│   ├── detail/          # Individual detail panel, generation rows
│   ├── layout/          # Sidebar navigation, theme toggle
│   └── problem-editor/  # Canvas editor + 5-step launch wizard
├── composables/         # useEChart, useResizable, useExperimentViewState, useTheme
├── pages/               # One file per top-level route
├── services/            # Import/export helpers, validators
├── types/               # TypeScript types (simlab.ts, problem.ts)
└── utils/               # Comparison quality indicators (comparisonMetrics.ts)
```

### Navigation

| Sidebar item | Route | Purpose |
|---|---|---|
| Dashboard | `/dashboard` | Live feed of running / queued / finished experiments |
| Campaigns | `/campaigns` | Group related experiments; launch from here |
| Experiments | `/experiments` | Full list across all campaigns; direct detail access |
| Problems | `/problems` | WSN problem editor (full-screen canvas) |
| Sources | `/sources` | Firmware source-code repositories |
| Compare | `/compare` | Side-by-side comparison of two finished experiments |

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
  elements are required.

**Saving problems:** use the *Problems* panel (toolbar button) to persist the
current design to the server. Saved problems can be loaded, renamed, or deleted.
The background image (floor plan, map) is stored alongside the draft.

**Importing / exporting:** the toolbar also supports JSON import/export for
version control or sharing configurations between users.

### 2. Create a campaign

Go to **Campaigns** and create a new campaign. A campaign is a named container
for a set of related experiments — for example, all runs that compare NSGA-II
vs NSGA-III on the same problem.

### 3. Launch an experiment (5-step wizard)

Open a campaign and click **Launch experiment**. The wizard walks through:

| Step | Name | What to configure |
|---|---|---|
| 1 | **Problem** | Select the saved WSN problem to optimise |
| 2 | **Experiment** | Algorithm (NSGA-II / NSGA-III / variants), source repository (firmware), population size, number of generations, crossover/mutation rates, NSGA-III divisions |
| 3 | **Simulation** | Simulation duration (seconds), random seeds |
| 4 | **Objectives** | One row per metric to optimise — name (e.g. `energy`, `latency`, `coverage`) and direction (`Minimize` / `Maximize`) |
| 5 | **Data conversion** | Column names in the Cooja CSV log, plus one aggregation rule (mean / max / min / last) per metric |

#### Tip — reusing seeds across experiments

In Step 3, use **↓ Export** to save the current seed list as `seeds.json`, then
**↑ Import** on any subsequent experiment to reproduce the exact same random
initialisation — critical for fair algorithm comparisons.

### 4. Monitor experiments

The **Experiment Detail** page (`/experiments/:id`) updates live while the job
runs. It shows:

- Status badge and progress bar (generations completed / total).
- **Parameters card** with the full algorithm and simulation configuration.
- **Pareto Front** chart — switchable between 2-D scatter (coloured by
  non-domination rank) and 3-D WebGL scatter (when ≥ 3 objectives).
- **HV & GD per generation** — hypervolume and generational distance tracked
  generation by generation.
- **Objectives evolution** — normalised best-per-generation line chart for each
  objective.
- **Parallel coordinates** — each Pareto solution as a poly-line across all
  objective axes (↑ = better for every axis); lines are coloured by the first
  objective's value.
- **Generations table** — expandable rows with per-individual chromosome,
  objectives, topology image, and simulations list.

Every chart card is **drag-resizable** (grab the bottom edge). Chart heights,
the 2-D / 3-D toggle, and axis selections are all **persisted per experiment**
in `sessionStorage`, so they survive navigation back to the page.

**Clicking a Pareto solution** in any chart (scatter or parallel coordinates)
opens the **Individual Detail** side panel: topology image, objective values,
chromosome key-value table, and the list of associated simulations.

### 5. Compare experiments

Go to **Compare** (`/compare`), select a campaign and two finished experiments,
and click **Compare**. The page shows:

- **Parameters table** — side-by-side diff of all algorithm / simulation parameters.
- **Pareto Front comparison** — both fronts overlaid on the same 2-D chart
  (axis-selectable) or 3-D WebGL chart.
- **Convergence chart** — hypervolume per generation for both experiments on the
  same axis.
- **Quality indicators** table:

  | Indicator | Meaning |
  |---|---|
  | C(A→B) | Coverage — fraction of B's solutions dominated by A |
  | C(B→A) | Coverage in the reverse direction |
  | ε-indicator (A→B) | Additive epsilon: how much A needs to shift to dominate every point in B |
  | IGD+ (A→B) | Inverted Generational Distance Plus — proximity + spread |
  | Spacing A / B | Schott's spacing — uniformity of distribution (lower = more uniform) |

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
- **Dominance mode** — click a point to highlight the region it dominates.

The camera position is preserved across data refreshes (polling) so the user's
viewpoint is not reset while the experiment runs.

### HV & GD per generation

Dual-axis line chart: hypervolume (higher = better) on the left axis, generational
distance (lower = better) on the right. One line per objective pair /
normalisation used.

### Objectives evolution

Normalised best-per-generation value for each objective, rescaled to [0 %, 100 %]
so objectives with different units can be compared on the same axis.

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
| `VITE_API_BASE_URL` | `http://localhost:8000/api/v1` | REST API base URL |
| `VITE_API_KEY` | `api-password` | `X-API-Key` header sent with every request |
