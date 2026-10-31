<template>
  <section class="metric">
    <header class="metric__head">
      <div>
        <h3 class="metric__title">{{ metric.name }}</h3>
        <p v-if="metric.aliases.length" class="metric__aliases">
          Same numbers as {{ metric.aliases.join(", ") }}.
        </p>
      </div>
      <label v-if="detail.panel.length > 1" class="chooser">
        Solution
        <select v-model="chromosome">
          <option v-for="(hash, i) in detail.panel" :key="hash" :value="hash">
            {{ chromosomeLabel(hash, i) }}
          </option>
        </select>
      </label>
    </header>

    <p v-if="error" class="metric__error" role="alert">{{ error }}</p>
    <p v-else-if="!data" class="muted">Loading…</p>
    <template v-else>
      <dl class="facts">
        <div>
          <dt>n</dt>
          <dd>{{ data.summary.n }}</dd>
        </div>
        <div>
          <dt>Mean</dt>
          <dd>{{ formatValue(data.summary.mean) }}</dd>
        </div>
        <div>
          <dt>{{ percent }}% CI (t)</dt>
          <dd>{{ formatInterval(data.intervals.mean_t) }}</dd>
        </div>
        <div>
          <dt>{{ percent }}% CI (BCa)</dt>
          <dd>{{ formatInterval(data.intervals.mean_bca) }}</dd>
        </div>
        <div>
          <dt>S</dt>
          <dd>
            {{ formatValue(data.summary.sd) }}
            <small>{{ formatInterval(data.intervals.sd_bca) }}</small>
          </dd>
        </div>
        <div>
          <dt>Median</dt>
          <dd>
            {{ formatValue(data.summary.median) }}
            <small>{{ formatInterval(data.intervals.median) }}</small>
          </dd>
        </div>
        <div>
          <dt>p5 / p95</dt>
          <dd>{{ formatValue(data.summary.p5) }} / {{ formatValue(data.summary.p95) }}</dd>
        </div>
        <div>
          <dt>Skewness / ex. kurtosis</dt>
          <dd>
            {{ formatValue(data.summary.skewness) }} /
            {{ formatValue(data.summary.excess_kurtosis) }}
          </dd>
        </div>
        <div v-if="data.intervals.p90_upper !== null">
          <dt>p90 upper limit</dt>
          <dd>{{ formatValue(data.intervals.p90_upper) }}</dd>
        </div>
        <div v-if="data.intervals.p95_upper !== null">
          <dt>p95 upper limit</dt>
          <dd>{{ formatValue(data.intervals.p95_upper) }}</dd>
        </div>
        <div v-if="data.n_star">
          <dt>n* (this solution)</dt>
          <dd>
            {{ data.n_star.value ?? "—" }}
            <small v-if="data.n_star.range">
              {{ data.n_star.range[0] }} to {{ data.n_star.range[1] }}
            </small>
          </dd>
        </div>
        <div v-if="data.n_star">
          <dt>Stein / resampling</dt>
          <dd>
            {{ data.n_star.stein ?? "—" }} / {{ data.precision?.n_emp ?? `> ${data.summary.n}` }}
          </dd>
        </div>
      </dl>

      <div class="tabs" role="tablist" aria-label="Distribution chart">
        <button
          v-for="tab in TABS"
          :key="tab.id"
          type="button"
          role="tab"
          :aria-selected="view === tab.id"
          :class="['tab', { active: view === tab.id }]"
          @click="view = tab.id"
        >
          {{ tab.label }}
        </button>
      </div>
      <SeedStudyChart :title="chartTitle" :name="`${metric.name}-${view}`" :build="build" />

      <details class="values">
        <summary>Values ({{ data.values.length }})</summary>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th scope="col">#</th>
                <th scope="col">Seed</th>
                <th scope="col">Value</th>
                <th scope="col">Flag</th>
                <th scope="col">Simulation</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="v in data.values" :key="v.seed" :class="{ flagged: v.flag }">
                <td>{{ v.order + 1 }}</td>
                <td class="mono">{{ v.seed }}</td>
                <td>{{ formatValue(v.value) }}</td>
                <td>{{ v.flag ? `outlier (${v.flag})` : "" }}</td>
                <td class="mono">
                  <RouterLink v-if="v.simulation_id" :to="`/experiments/${detail.id}`">
                    {{ v.simulation_id.slice(-8) }}
                  </RouterLink>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <p class="note">Outliers are marked and stay in every calculation.</p>
      </details>
    </template>

    <template v-if="metric.panel">
      <h4 class="metric__sub">Across solutions</h4>
      <dl class="facts">
        <div>
          <dt>Between solutions</dt>
          <dd>{{ formatPercent(metric.panel.shares.individual) }}</dd>
        </div>
        <div>
          <dt>Seed alone</dt>
          <dd>{{ formatPercent(metric.panel.shares.seed) }}</dd>
        </div>
        <div>
          <dt>Seed × solution</dt>
          <dd>{{ formatPercent(metric.panel.shares.residual) }}</dd>
        </div>
        <div>
          <dt>m* (Eρ² ≥ {{ reliability }})</dt>
          <dd>
            {{ metric.panel.m_star.value ?? "—" }}
            <small v-if="metric.panel.m_star.ci">
              {{ metric.panel.m_star.ci[0] ?? "—" }} to {{ metric.panel.m_star.ci[1] ?? "∞" }}
            </small>
          </dd>
        </div>
        <div>
          <dt>By ranking, disjoint seeds</dt>
          <dd>{{ metric.panel.m_rank ?? "—" }}</dd>
        </div>
        <div>
          <dt>Panel used</dt>
          <dd>{{ metric.panel.used.chromosomes }} × {{ metric.panel.used.seeds }}</dd>
        </div>
      </dl>
      <div class="pair">
        <SeedStudyChart
          title="Reliability by number of seeds"
          :name="`${metric.name}-reliability`"
          :build="(c) => reliabilityOption(metric.panel!, reliability, c)"
        />
        <SeedStudyChart
          title="Each solution across seeds (p5 to p95)"
          :name="`${metric.name}-solutions`"
          :build="(c) => bySolutionOption(metric, c)"
        />
      </div>
    </template>
  </section>
</template>

<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { RouterLink } from "vue-router";
import type { EChartsOption } from "echarts";
import { getStudyMetric } from "../../api/seedStudy";
import type { chartPalette } from "../../services/chartTheme";
import {
  chromosomeLabel,
  formatInterval,
  formatPercent,
  formatValue,
} from "../../services/seedStudy";
import {
  boxOption,
  bySolutionOption,
  ecdfOption,
  histogramOption,
  orderOption,
  precisionOption,
  qqOption,
  reliabilityOption,
  runningOption,
} from "../../services/seedStudyCharts";
import type { MetricDetail, StudyDetail, StudyMetric, StudyTargets } from "../../types/seedStudy";
import SeedStudyChart from "./SeedStudyChart.vue";

const TABS = [
  { id: "histogram", label: "Histogram" },
  { id: "ecdf", label: "ECDF" },
  { id: "box", label: "Box" },
  { id: "qq", label: "Q-Q" },
  { id: "order", label: "By run order" },
  { id: "running", label: "Running mean" },
  { id: "precision", label: "Precision" },
] as const;

type Tab = (typeof TABS)[number]["id"];
type Palette = ReturnType<typeof chartPalette>;

const props = defineProps<{
  detail: StudyDetail;
  metric: StudyMetric;
  targets: StudyTargets;
}>();

const chromosome = ref(props.detail.panel[0] ?? "");
const view = ref<Tab>("histogram");
const data = ref<MetricDetail | null>(null);
const error = ref("");

const percent = computed(() => Math.round(props.targets.confidence * 1000) / 10);
const reliability = computed(() => props.targets.reliability);
const noTarget = computed(() => data.value?.n_star?.method === "none");

const chartTitle = computed(() => {
  if (view.value === "precision") {
    return noTarget.value
      ? "The target moves with the number of seeds"
      : `Chance of landing within ±${props.targets.relative_error * 100}%, by number of seeds`;
  }
  return TABS.find((tab) => tab.id === view.value)!.label;
});

function build(c: Palette): EChartsOption {
  const current = data.value!;
  switch (view.value) {
    case "ecdf":
      return ecdfOption(current, c);
    case "box":
      return boxOption(current, c);
    case "qq":
      return qqOption(current, c);
    case "order":
      return orderOption(current, c);
    case "running":
      return runningOption(current, c);
    case "precision":
      return precisionOption(current, props.targets.confidence, noTarget.value, c);
    default:
      return histogramOption(current, c);
  }
}

let inFlight: AbortController | null = null;

async function load() {
  inFlight?.abort();
  const controller = new AbortController();
  inFlight = controller;
  error.value = "";
  try {
    const next = await getStudyMetric(
      props.detail.id,
      props.metric.name,
      chromosome.value,
      props.targets,
      controller.signal,
    );
    if (inFlight === controller) data.value = next;
  } catch (e) {
    if (!controller.signal.aborted) {
      data.value = null;
      error.value = e instanceof Error ? e.message : String(e);
    }
  }
}

watch(
  () => [
    props.detail.id,
    props.metric.name,
    chromosome.value,
    props.detail.finished,
    props.targets.confidence,
    props.targets.relative_error,
  ],
  load,
  { immediate: true },
);

watch(
  () => props.detail.id,
  () => (chromosome.value = props.detail.panel[0] ?? ""),
);
</script>

<style scoped>
.metric {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.metric__head {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  gap: 12px;
  flex-wrap: wrap;
}

.metric__title {
  margin: 0;
  font-size: 15px;
  font-weight: 700;
}

.metric__aliases,
.note {
  margin: 2px 0 0;
  font-size: 12px;
  color: var(--color-text-muted);
}

.metric__sub {
  margin: 8px 0 0;
  font-size: 13px;
  font-weight: 700;
}

.metric__error {
  margin: 0;
  color: var(--status-error);
  font-size: 13px;
}

.chooser {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: var(--color-text-muted);
}

.chooser select {
  padding: 5px 8px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: var(--color-bg);
  color: var(--color-text);
  font-size: 12px;
}

.facts {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 8px;
  margin: 0;
}

.facts div {
  padding: 8px 10px;
  background: var(--color-bg);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  min-width: 0;
}

.facts dt {
  font-size: 11px;
  color: var(--color-text-muted);
}

.facts dd {
  margin: 2px 0 0;
  font-size: 13px;
  font-weight: 600;
  overflow-wrap: anywhere;
}

.facts small {
  display: block;
  font-weight: 400;
  color: var(--color-text-muted);
}

.tabs {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}

.tab {
  padding: 5px 12px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: var(--color-bg);
  color: var(--color-text-muted);
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
}

.tab.active {
  border-color: var(--color-primary);
  color: var(--color-link, var(--color-primary));
  background: var(--color-primary-light);
}

.pair {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 16px;
}

.values summary {
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
}

.table-wrap {
  overflow-x: auto;
  max-height: 340px;
  margin-top: 8px;
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}

th,
td {
  padding: 5px 8px;
  border-bottom: 1px solid var(--color-border);
  text-align: left;
}

th {
  color: var(--color-text-muted);
}

tr.flagged td {
  background: var(--color-primary-light);
}

.mono {
  font-family: var(--font-mono, ui-monospace, monospace);
}

.muted {
  color: var(--color-text-muted);
  font-size: 13px;
  font-style: italic;
}
</style>
