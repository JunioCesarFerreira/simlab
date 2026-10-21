<template>
  <section class="result">
    <header class="result__head">
      <div>
        <h2 class="result__title">{{ detail.name }}</h2>
        <p class="result__meta">
          {{ detail.problem || "—" }} · {{ detail.duration }} s simulated ·
          {{ detail.waves_finished ?? 0 }}/{{ detail.waves }} waves ·
          <RouterLink :to="`/experiments/${detail.id}`">raw generations ↗</RouterLink>
        </p>
        <p v-if="detail.system_message" :class="['result__msg', `result__msg--${tone}`]">
          {{ detail.system_message }}
        </p>
      </div>
      <div class="metric-toggle" role="group" aria-label="Throughput definition">
        <button
          v-for="m in analysis.metrics"
          :key="m"
          type="button"
          :class="['metric-btn', { active: metric === m }]"
          :aria-pressed="metric === m"
          :title="METRIC_LABELS[m].hint"
          @click="metric = m"
        >
          {{ METRIC_LABELS[m].label }}
        </button>
      </div>
    </header>

    <p class="metric-hint">{{ METRIC_LABELS[metric].hint }}</p>

    <div class="tiles">
      <div class="tile">
        <div class="tile__value">{{ formatThroughput(fit.lam) }}</div>
        <div class="tile__label">λ — one simulation at a time (sim/min)</div>
      </div>
      <div class="tile">
        <div class="tile__value">{{ formatCoefficient(fit.sigma) }}</div>
        <div class="tile__label">σ — contention</div>
      </div>
      <div class="tile">
        <div class="tile__value">{{ formatCoefficient(fit.kappa) }}</div>
        <div class="tile__label">κ — coherency</div>
      </div>
      <div :class="['tile', { 'tile--peak': verdictValue === 'retrograde' }]">
        <div class="tile__value">
          {{ fit.n_peak_int ?? "—" }}
          <span v-if="fit.n_peak_int && !fit.peak_within_tested" class="tile__sub">extrap.</span>
        </div>
        <div class="tile__label">
          N* — peak
          <template v-if="fit.x_peak">({{ formatThroughput(fit.x_peak) }} sim/min)</template>
          <template v-else-if="fit.x_asymptote">
            — none; ceiling {{ formatThroughput(fit.x_asymptote) }}</template
          >
        </div>
      </div>
      <div class="tile">
        <div class="tile__value">{{ fit.r2 === null ? "—" : fit.r2.toFixed(3) }}</div>
        <div class="tile__label">R² over {{ fit.n_points }} wave(s)</div>
      </div>
    </div>

    <ul class="reading">
      <li v-for="(line, i) in reading" :key="i">{{ line }}</li>
    </ul>

    <UslChart :data="chart" :name="detail.name" />

    <details class="waves" :open="analysis.waves.length <= 12">
      <summary>Waves ({{ analysis.valid_waves }} of {{ analysis.waves.length }} measured)</summary>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th scope="col">#</th>
              <th scope="col">N</th>
              <th scope="col">Rep.</th>
              <th scope="col">Makespan</th>
              <th scope="col">Mean held</th>
              <th scope="col">X ({{ METRIC_LABELS[metric].label.toLowerCase() }})</th>
              <th scope="col">Hosts</th>
              <th scope="col">Note</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="w in analysis.waves" :key="w.generation_id" :class="{ invalid: !w.valid }">
              <td>{{ (w.wave ?? 0) + 1 }}</td>
              <td>{{ w.level }}</td>
              <td>{{ (w.repetition ?? 0) + 1 }}</td>
              <td>{{ formatSeconds(w.makespan_sec) }}</td>
              <td>{{ formatSeconds(w.mean_held_sec) }}</td>
              <td>{{ formatThroughput(w.throughput[metric]) }}</td>
              <td class="hosts">{{ hostsLabel(w.hosts) }}</td>
              <td class="note">{{ w.valid ? "" : w.reason }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </details>
  </section>
</template>

<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { RouterLink } from "vue-router";
import UslChart from "./UslChart.vue";
import {
  METRIC_LABELS,
  chartData,
  formatCoefficient,
  formatSeconds,
  formatThroughput,
  interpret,
  verdict,
} from "../../services/usl";
import type { BenchmarkDetail, UslMetric } from "../../types/scalability";

const props = defineProps<{ detail: BenchmarkDetail }>();

const analysis = computed(() => props.detail.analysis);
const metric = ref<UslMetric>(props.detail.analysis.default_metric);
// A different benchmark selected: start from its default again.
watch(
  () => props.detail.id,
  () => (metric.value = props.detail.analysis.default_metric),
);

const fit = computed(() => analysis.value.fits[metric.value]);
const testedMax = computed(() => Math.max(1, ...analysis.value.waves.map((w) => w.level)));
const chart = computed(() => chartData(fit.value, analysis.value.waves, metric.value));
const reading = computed(() => interpret(fit.value, testedMax.value));
const verdictValue = computed(() => verdict(fit.value));
const tone = computed(() =>
  props.detail.status === "Error" ? "error" : props.detail.status === "Done" ? "ok" : "muted",
);

function hostsLabel(hosts: Record<string, number>): string {
  return Object.entries(hosts)
    .map(([host, n]) => `${host} ×${n}`)
    .join(", ");
}
</script>

<style scoped>
.result {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.result__head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 12px;
  flex-wrap: wrap;
}

.result__title {
  font-size: 17px;
  font-weight: 700;
  margin: 0;
}

.result__meta,
.metric-hint {
  margin: 2px 0 0;
  font-size: 12px;
  color: var(--color-text-muted);
}

.result__msg {
  margin: 6px 0 0;
  font-size: 12px;
}

.result__msg--error {
  color: var(--status-error);
}

.result__msg--ok {
  color: var(--status-done);
}

.result__msg--muted {
  color: var(--color-text-muted);
}

.metric-toggle {
  display: inline-flex;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  overflow: hidden;
}

.metric-btn {
  padding: 6px 12px;
  font-size: 12px;
  font-weight: 600;
  background: var(--color-bg);
  color: var(--color-text-muted);
  border: none;
  cursor: pointer;
}

.metric-btn + .metric-btn {
  border-left: 1px solid var(--color-border);
}

.metric-btn.active {
  background: var(--color-primary);
  color: #fff;
}

.tiles {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 10px;
}

.tile {
  background: var(--color-bg);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  padding: 12px 14px;
}

.tile--peak {
  border-color: var(--status-error);
}

.tile__value {
  font-size: 22px;
  font-weight: 800;
  line-height: 1.1;
  font-variant-numeric: tabular-nums;
}

.tile__sub {
  font-size: 11px;
  font-weight: 600;
  color: var(--color-text-muted);
}

.tile__label {
  margin-top: 4px;
  font-size: 11px;
  color: var(--color-text-muted);
}

.reading {
  margin: 0;
  padding-left: 18px;
  font-size: 13px;
  line-height: 1.55;
}

.waves summary {
  cursor: pointer;
  font-size: 12px;
  font-weight: 600;
  color: var(--color-text-muted);
  padding: 4px 0;
}

.table-wrap {
  overflow-x: auto;
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
  font-variant-numeric: tabular-nums;
}

th,
td {
  text-align: left;
  padding: 6px 8px;
  border-bottom: 1px solid var(--color-border);
  white-space: nowrap;
}

th {
  font-weight: 600;
  color: var(--color-text-muted);
}

tr.invalid td {
  color: var(--color-text-muted);
}

td.note {
  white-space: normal;
  min-width: 200px;
}

td.hosts {
  white-space: normal;
}
</style>
