<template>
  <section class="result">
    <header class="result__head">
      <div>
        <h2 class="result__title">{{ detail.name }}</h2>
        <p class="result__meta">
          {{ detail.problem || "—" }} · {{ detail.duration }} min simulated ·
          {{ detail.chromosomes }} solution(s) × {{ detail.seeds }} seeds ·
          {{ detail.finished ?? 0 }}/{{ detail.simulations }} simulations ·
          <RouterLink :to="`/experiments/${detail.id}`">raw generations ↗</RouterLink>
        </p>
        <p v-if="detail.system_message" :class="['result__msg', `result__msg--${tone}`]">
          {{ detail.system_message }}
        </p>
      </div>
      <div class="exports">
        <button type="button" class="ghost-btn" :disabled="!analysis" @click="downloadCsv">
          CSV
        </button>
        <button type="button" class="ghost-btn" :disabled="!analysis" @click="downloadJson">
          JSON
        </button>
        <button type="button" class="ghost-btn" @click="downloadSeeds">seeds.json</button>
      </div>
    </header>

    <ul v-if="detail.top_warnings.length" class="alerts">
      <li v-for="w in detail.top_warnings" :key="w.code" class="alert alert--bad">
        {{ w.message }}
      </li>
    </ul>

    <div class="checks">
      <span :class="['check', `check--${controlTone}`]">{{ controlLabel }}</span>
      <span class="check">
        {{ detail.completeness.observed }}/{{ detail.completeness.planned }} results
      </span>
      <span class="check" :title="failureHint">
        {{ detail.completeness.failures.failed }} failed attempt(s)
      </span>
      <span v-for="image in detail.provenance.images" :key="image" class="check">{{ image }}</span>
    </div>

    <template v-if="analysis">
      <section class="answer" aria-label="How many seeds">
        <h3 class="answer__title">How many seeds?</h3>
        <div class="answer__columns">
          <div class="answer__column">
            <h4>To rank solutions (optimisation)</h4>
            <p class="answer__hint">Reliability of {{ targets.reliability }}.</p>
            <ul class="numbers">
              <li v-for="(row, name) in recommendation.optimise.by_objective" :key="name">
                <span class="numbers__name">{{ name }}</span>
                <strong>{{ row.value ?? "—" }}</strong>
                <small v-if="row.ci"> {{ row.ci[0] ?? "—" }} to {{ row.ci[1] ?? "∞" }} </small>
              </li>
            </ul>
          </div>
          <div class="answer__column">
            <h4>To report one solution's value</h4>
            <p class="answer__hint">
              ±{{ percent(targets.relative_error) }}, {{ percent(targets.confidence) }}
              confidence. Median across the panel.
            </p>
            <ul class="numbers">
              <li v-for="(row, name) in recommendation.report.by_objective" :key="name">
                <span class="numbers__name">{{ name }}</span>
                <strong>{{ row.value ?? "—" }}</strong>
                <small v-if="row.p90 !== null && detail.chromosomes > 1"> p90 {{ row.p90 }} </small>
                <small v-if="row.status !== 'ok'" :title="statusInfo(row.status).hint">
                  {{ statusInfo(row.status).label }}
                </small>
              </li>
            </ul>
            <p v-if="recommendation.report.joint" class="answer__hint">
              All together, {{ recommendation.report.joint.value }} seeds at
              {{ percent(recommendation.report.joint.confidence) }} (Bonferroni).
            </p>
          </div>
        </div>
        <p class="answer__headline">
          <strong>{{ headline(detail) }}</strong> {{ scopeSentence(detail) }}
        </p>
        <p v-if="!detail.synthetic" class="answer__note">
          This radio model has no random loss: the seed moves boot times, jitter and backoff. A low
          variation here does not mean a real network would be as stable.
        </p>
      </section>

      <div class="targets">
        <label>
          Confidence (%)
          <input v-model.number="confidencePct" type="number" min="80" max="99.9" step="0.5" />
        </label>
        <label>
          Relative error (%)
          <input v-model.number="errorPct" type="number" min="0.1" max="100" step="0.5" />
        </label>
        <label>
          Reliability
          <input v-model.number="reliabilityValue" type="number" min="0.5" max="0.99" step="0.01" />
        </label>
        <small>Recalculated from the same runs. Nothing is simulated again.</small>
      </div>

      <details v-if="analysis.warnings.length" class="advisories" open>
        <summary>Notes on this analysis ({{ analysis.warnings.length }})</summary>
        <ul class="alerts">
          <li v-for="(w, i) in analysis.warnings" :key="i" class="alert alert--warn">
            {{ w.message }}
          </li>
        </ul>
      </details>

      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th scope="col">Metric</th>
              <th scope="col">Results</th>
              <th scope="col">{{ detail.chromosomes > 1 ? "Median of means" : "Mean" }}</th>
              <th scope="col">CV</th>
              <th scope="col">{{ detail.chromosomes > 1 ? "CI (first solution)" : "CI" }}</th>
              <th scope="col" title="Shapiro–Wilk with Holm's correction. A diagnostic only.">
                Normal?
              </th>
              <th scope="col" title="Seeds to estimate one solution">n*</th>
              <th v-if="detail.chromosomes > 1" scope="col" title="Seeds to rank solutions">m*</th>
              <th scope="col">Status</th>
            </tr>
          </thead>
          <tbody>
            <tr
              v-for="m in analysis.metrics"
              :key="m.name"
              :class="['row', { selected: m.name === selected?.name, faded: m.status !== 'ok' }]"
              tabindex="0"
              @click="select(m)"
              @keydown.enter="select(m)"
            >
              <td class="name">
                {{ m.name }}
                <small v-if="m.is_objective">
                  objective · {{ m.goal }} · {{ aggregatorLabel(m.aggregator) }}
                </small>
                <small v-if="m.aliases.length">also {{ m.aliases.join(", ") }}</small>
              </td>
              <td>{{ m.observed }}/{{ m.planned }}</td>
              <td>{{ formatValue(centre(m)) }}</td>
              <td>{{ m.summary.cv ? formatPercent(m.summary.cv.p50) : "—" }}</td>
              <td>{{ formatInterval(m.by_chromosome[0]?.ci_mean_t) }}</td>
              <td>{{ normalityLabel(m.summary.normality_rejected) }}</td>
              <td>{{ m.summary.n_star ? Math.ceil(m.summary.n_star.p50) : "—" }}</td>
              <td v-if="detail.chromosomes > 1">{{ m.panel?.m_star.value ?? "—" }}</td>
              <td>
                <span :class="['badge', `badge--${statusOf(m).tone}`]" :title="statusOf(m).hint">
                  {{ m.all_zero ? "always zero" : statusOf(m).label }}
                </span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <SeedStudyMetric
        v-if="selected && selected.status !== 'no_data'"
        :key="selected.name"
        :detail="detail"
        :metric="selected"
        :targets="targets"
      />

      <details v-if="analysis.front_stability" class="advisories">
        <summary>Pareto front between disjoint seed sets</summary>
        <p class="note">
          Share of the front two disjoint sets of seeds agree on. A diagnostic, not a stopping rule:
          with many near-ties, membership keeps changing after the ranking has settled.
        </p>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th scope="col">Seeds per set</th>
                <th v-for="row in analysis.front_stability" :key="row.k" scope="col">
                  {{ row.k }}
                </th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Jaccard</td>
                <td v-for="row in analysis.front_stability" :key="row.k">
                  {{ row.jaccard.toFixed(2) }}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </details>

      <details v-if="detail.completeness.failed_total" class="advisories">
        <summary>Seeds with no result ({{ detail.completeness.failed_total }})</summary>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th scope="col">Solution</th>
                <th scope="col">Seed</th>
                <th scope="col">Cause</th>
                <th scope="col">Message</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="f in detail.completeness.failed_seeds" :key="`${f.genome_hash}-${f.seed}`">
                <td class="mono">{{ f.genome_hash.slice(0, 8) }}</td>
                <td class="mono">{{ f.seed }}</td>
                <td>{{ f.kind }}</td>
                <td>{{ f.message }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </details>
    </template>
    <p v-else class="muted">No result yet. The analysis fills in as simulations finish.</p>
  </section>
</template>

<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { RouterLink } from "vue-router";
import { getStudyCsv } from "../../api/seedStudy";
import { serializeSeedFile } from "../../lib/seedFile";
import {
  aggregatorLabel,
  formatInterval,
  formatPercent,
  formatValue,
  headline,
  metricStatus,
  normalityLabel,
  scopeSentence,
  statusInfo,
} from "../../services/seedStudy";
import type { StudyDetail, StudyMetric, StudyTargets } from "../../types/seedStudy";
import SeedStudyMetric from "./SeedStudyMetric.vue";

const props = defineProps<{ detail: StudyDetail }>();
const emit = defineEmits<{ targets: [StudyTargets] }>();

const analysis = computed(() => props.detail.analysis);
const recommendation = computed(() => props.detail.analysis!.recommendation);
const targets = computed(() => props.detail.analysis?.targets ?? props.detail.targets);
const selectedName = ref<string | null>(null);

const selected = computed(() => {
  const metrics = analysis.value?.metrics ?? [];
  return metrics.find((m) => m.name === selectedName.value) ?? metrics[0] ?? null;
});

const confidencePct = ref(0);
const errorPct = ref(0);
const reliabilityValue = ref(0);

function syncTargets() {
  confidencePct.value = Math.round(targets.value.confidence * 1000) / 10;
  errorPct.value = Math.round(targets.value.relative_error * 1000) / 10;
  reliabilityValue.value = targets.value.reliability;
}
syncTargets();
watch(() => props.detail.id, syncTargets);

let debounce: ReturnType<typeof setTimeout> | null = null;

watch([confidencePct, errorPct, reliabilityValue], ([confidence, error, reliability]) => {
  const valid =
    confidence >= 80 &&
    confidence <= 99.9 &&
    error >= 0.1 &&
    error <= 100 &&
    reliability >= 0.5 &&
    reliability <= 0.99;
  if (!valid) return;
  if (debounce) clearTimeout(debounce);
  debounce = setTimeout(() => {
    const next = {
      confidence: confidence / 100,
      relative_error: error / 100,
      reliability,
    };
    const current = targets.value;
    if (
      next.confidence !== current.confidence ||
      next.relative_error !== current.relative_error ||
      next.reliability !== current.reliability
    ) {
      emit("targets", next);
    }
  }, 350);
});

const tone = computed(() =>
  props.detail.status === "Error" ? "error" : props.detail.status === "Done" ? "ok" : "muted",
);

const controlTone = computed(() => {
  const control = props.detail.determinism;
  if (!control.checked) return "muted";
  return control.identical ? "ok" : "bad";
});

const controlLabel = computed(() => {
  const control = props.detail.determinism;
  if (!control.checked) return "control pending";
  return control.identical
    ? `control identical (${control.compared} comparisons)`
    : `control diverged (${control.divergent_total})`;
});

const failureHint = computed(() => {
  const { failures } = props.detail.completeness;
  const interval = failures.rate_ci;
  return (
    `${failures.by_kind.infrastructure} infrastructure, ${failures.by_kind.execution} execution` +
    (interval
      ? `. Rate within ${formatPercent(interval[0])} to ${formatPercent(interval[1])}.`
      : "")
  );
});

function percent(value: number): string {
  return `${Math.round(value * 1000) / 10}%`;
}

function select(metric: StudyMetric) {
  selectedName.value = metric.name;
}

function centre(metric: StudyMetric): number | null {
  const means = metric.by_chromosome
    .map((row) => row.mean)
    .filter((mean): mean is number => mean !== null)
    .sort((a, b) => a - b);
  return means.length ? (means[Math.floor((means.length - 1) / 2)] ?? null) : null;
}

function statusOf(metric: StudyMetric) {
  return statusInfo(
    metricStatus(metric, analysis.value?.recommendation.report.by_objective[metric.name]),
  );
}

function save(content: string, filename: string, type: string) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

async function downloadCsv() {
  save(await getStudyCsv(props.detail.id), `seed-study-${props.detail.id}.csv`, "text/csv");
}

function downloadJson() {
  save(
    JSON.stringify(props.detail, null, 2),
    `seed-study-${props.detail.id}.json`,
    "application/json",
  );
}

function downloadSeeds() {
  save(serializeSeedFile(props.detail.seed_list), "seeds.json", "application/json");
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
.note {
  margin: 4px 0 0;
  font-size: 12px;
  color: var(--color-text-muted);
}

.result__msg {
  margin: 4px 0 0;
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

.exports {
  display: flex;
  gap: 6px;
}

.ghost-btn {
  padding: 5px 12px;
  border-radius: var(--radius-md);
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  background: var(--color-bg);
  color: var(--color-text);
  border: 1px solid var(--color-border);
}

.ghost-btn:disabled {
  opacity: 0.5;
  cursor: default;
}

.alerts {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.alert {
  padding: 8px 12px;
  border-radius: var(--radius-md);
  font-size: 13px;
  line-height: 1.45;
}

.alert--bad {
  background: #fee2e2;
  color: #991b1b;
  border: 1px solid #fecaca;
}

.alert--warn {
  background: #fef3c7;
  color: #92400e;
  border: 1px solid #fde68a;
}

.checks {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.check {
  font-size: 11px;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 999px;
  border: 1px solid var(--color-border);
  color: var(--color-text-muted);
}

.check--ok {
  color: var(--status-done);
}

.check--bad {
  color: var(--status-error);
}

.answer {
  padding: 14px 16px;
  border: 1px solid var(--color-primary);
  border-radius: var(--radius-lg);
  background: var(--color-bg);
}

.answer__title {
  margin: 0 0 10px;
  font-size: 14px;
  font-weight: 700;
}

.answer__columns {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 16px;
}

.answer__column h4 {
  margin: 0;
  font-size: 13px;
  font-weight: 700;
}

.answer__hint {
  margin: 2px 0 8px;
  font-size: 12px;
  color: var(--color-text-muted);
}

.numbers {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.numbers li {
  display: flex;
  flex-direction: column;
  min-width: 96px;
  padding: 6px 10px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: var(--color-surface);
}

.numbers strong {
  font-size: 22px;
  font-weight: 700;
  line-height: 1.2;
}

.numbers__name,
.numbers small {
  font-size: 11px;
  color: var(--color-text-muted);
}

.answer__headline {
  margin: 12px 0 0;
  font-size: 13px;
}

.answer__note {
  margin: 4px 0 0;
  font-size: 12px;
  color: var(--color-text-muted);
}

.targets {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-end;
  gap: 12px;
}

.targets label {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 12px;
  color: var(--color-text-muted);
}

.targets input {
  width: 110px;
  padding: 5px 8px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: var(--color-bg);
  color: var(--color-text);
  font-size: 13px;
}

.targets small {
  font-size: 11px;
  color: var(--color-text-muted);
}

.advisories summary {
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  margin-bottom: 6px;
}

.table-wrap {
  overflow-x: auto;
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}

th,
td {
  padding: 6px 8px;
  border-bottom: 1px solid var(--color-border);
  text-align: left;
  white-space: nowrap;
}

th {
  color: var(--color-text-muted);
  font-weight: 600;
}

.row {
  cursor: pointer;
}

.row:hover td {
  background: var(--color-bg);
}

.row.selected td {
  background: var(--color-primary-light);
}

.row.faded .name {
  color: var(--color-text-muted);
}

.name {
  font-weight: 600;
}

.name small {
  display: block;
  font-weight: 400;
  color: var(--color-text-muted);
}

.badge {
  font-size: 11px;
  font-weight: 600;
  padding: 1px 8px;
  border-radius: 999px;
  border: 1px solid var(--color-border);
}

.badge--ok {
  color: var(--status-done);
}

.badge--warn {
  color: #b45309;
}

.badge--bad {
  color: var(--status-error);
}

.badge--muted {
  color: var(--color-text-muted);
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
