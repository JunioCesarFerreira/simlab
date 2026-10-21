<template>
  <div class="scal">
    <div class="page-header">
      <div>
        <h1 class="page-title">Scalability</h1>
        <p class="subtitle">
          Run one simulation at increasing concurrency and fit the Universal Scalability Law to the
          time each wave takes: contention σ, coherency κ, and where throughput peaks.
        </p>
      </div>
      <div class="header-actions">
        <button class="refresh-btn" :disabled="loading" @click="load()">
          {{ loading ? "Loading…" : "Refresh" }}
        </button>
        <button class="primary-btn" @click="showForm = !showForm">
          {{ showForm ? "Close" : "New benchmark" }}
        </button>
      </div>
    </div>

    <p v-if="error" class="error-banner">{{ error }}</p>
    <p v-if="notice" class="notice-banner">{{ notice }}</p>

    <section v-if="showForm" class="card">
      <h2 class="card-title">New benchmark</h2>
      <BenchmarkForm :pool="pool" @created="onCreated" />
    </section>

    <section class="card">
      <h2 class="card-title">Benchmarks</h2>
      <div v-if="!benchmarks.length && !loading" class="empty-state">
        No benchmark yet. Start one to measure how this pool scales.
      </div>
      <ul v-else class="bench-list">
        <li v-for="b in benchmarks" :key="b.id">
          <button
            type="button"
            :class="['bench-row', { selected: b.id === selectedId }]"
            :aria-pressed="b.id === selectedId"
            @click="select(b.id)"
          >
            <span class="bench-row__name">{{ b.name }}</span>
            <span :class="['status', `status--${b.status.toLowerCase()}`]">{{ b.status }}</span>
            <span class="bench-row__plan">
              N = {{ b.levels.join(", ") }} · ×{{ b.repetitions }} · {{ b.simulations }} sims
            </span>
            <span
              class="bar"
              role="progressbar"
              :aria-valuenow="b.waves_finished ?? 0"
              aria-valuemin="0"
              :aria-valuemax="b.waves"
              :aria-label="`${b.waves_finished ?? 0} of ${b.waves} waves finished`"
            >
              <span class="bar__fill" :style="{ width: `${progress(b) * 100}%` }" />
            </span>
            <span class="bench-row__when">{{ when(b) }}</span>
          </button>
        </li>
      </ul>
    </section>

    <section v-if="detail" class="card">
      <BenchmarkResult :detail="detail" />
    </section>
    <p v-else-if="selectedId && detailLoading" class="muted">Loading benchmark…</p>
  </div>
</template>

<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from "vue";
import { useRoute, useRouter } from "vue-router";
import BenchmarkForm from "../components/scalability/BenchmarkForm.vue";
import BenchmarkResult from "../components/scalability/BenchmarkResult.vue";
import { getOverview } from "../api/infra";
import { getBenchmark, listBenchmarks } from "../api/scalability";
import { progress, schedulablePool } from "../services/usl";
import type { BenchmarkCreated, BenchmarkDetail, BenchmarkSummary } from "../types/scalability";

/** How often a running benchmark is re-read. Each wave takes one simulation's
 *  duration, so anything faster only repeats the same answer. */
const POLL_MS = 10_000;

const route = useRoute();
const router = useRouter();

const benchmarks = ref<BenchmarkSummary[]>([]);
const detail = ref<BenchmarkDetail | null>(null);
const selectedId = ref<string | null>((route.query.id as string) || null);
const pool = ref(0);
const loading = ref(false);
const detailLoading = ref(false);
const showForm = ref(false);
const error = ref("");
const notice = ref("");

let poll: ReturnType<typeof setInterval> | null = null;
let inFlight: AbortController | null = null;

function describe(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

function running(status?: string): boolean {
  return status === "Running" || status === "Waiting";
}

async function load(showSpinner = true) {
  if (showSpinner) loading.value = true;
  inFlight?.abort();
  const controller = new AbortController();
  inFlight = controller;
  try {
    const [list, overview] = await Promise.all([
      listBenchmarks(controller.signal),
      getOverview(controller.signal),
    ]);
    benchmarks.value = list;
    pool.value = schedulablePool(overview);
    const newest = list[0];
    if (!selectedId.value && newest) selectedId.value = newest.id;
    if (selectedId.value) await loadDetail(controller.signal);
    error.value = "";
  } catch (e) {
    if (!controller.signal.aborted) error.value = describe(e);
  } finally {
    if (inFlight === controller) inFlight = null;
    loading.value = false;
  }
}

async function loadDetail(signal?: AbortSignal) {
  if (!selectedId.value) return;
  detailLoading.value = true;
  try {
    detail.value = await getBenchmark(selectedId.value, signal);
  } finally {
    detailLoading.value = false;
  }
}

async function select(id: string) {
  selectedId.value = id;
  detail.value = null;
  router.replace({ query: { ...route.query, id } });
  try {
    await loadDetail();
  } catch (e) {
    error.value = describe(e);
  }
}

async function onCreated(created: BenchmarkCreated) {
  showForm.value = false;
  notice.value =
    `Benchmark queued: ${created.waves} wave(s), ${created.simulations} simulation(s). ` +
    "The curve fills in as waves finish.";
  selectedId.value = created.id;
  await load();
}

function when(b: BenchmarkSummary): string {
  const stamp = b.start_time || b.created_time;
  return stamp ? new Date(stamp).toLocaleString() : "";
}

onMounted(async () => {
  await load();
  // Re-read only while something is moving: a finished benchmark does not
  // change, and polling it would only repeat the same analysis.
  poll = setInterval(() => {
    const active = benchmarks.value.some((b) => running(b.status));
    if (active && !showForm.value) load(false);
  }, POLL_MS);
});

onBeforeUnmount(() => {
  if (poll) clearInterval(poll);
  inFlight?.abort();
});
</script>

<style scoped>
.scal {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.page-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  flex-wrap: wrap;
}

.page-title {
  font-size: 22px;
  font-weight: 700;
}

.subtitle {
  font-size: 13px;
  color: var(--color-text-muted);
  margin-top: 2px;
  max-width: 72ch;
  line-height: 1.5;
}

.header-actions {
  display: flex;
  align-items: center;
  gap: 8px;
}

.refresh-btn,
.primary-btn {
  padding: 7px 16px;
  border-radius: var(--radius-md);
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
}

.refresh-btn {
  background: var(--color-primary-light);
  color: var(--color-link, var(--color-primary));
  border: 1px solid #bfdbfe;
}

.primary-btn {
  background: var(--color-primary);
  color: #fff;
  border: 1px solid var(--color-primary);
}

.refresh-btn:disabled {
  opacity: 0.5;
  cursor: default;
}

.error-banner,
.notice-banner {
  padding: 12px 16px;
  border-radius: var(--radius-md);
  font-size: 13px;
  line-height: 1.5;
}

.error-banner {
  background: #fee2e2;
  color: var(--status-error);
  border: 1px solid #fecaca;
}

.notice-banner {
  background: var(--color-primary-light);
  color: var(--color-link, var(--color-primary));
  border: 1px solid #bfdbfe;
}

.card {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  padding: 18px 20px;
  box-shadow: var(--shadow-sm);
}

.card-title {
  font-size: 14px;
  font-weight: 700;
  margin: 0 0 12px;
}

.bench-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.bench-row {
  width: 100%;
  display: grid;
  grid-template-columns: minmax(140px, 1.4fr) auto minmax(160px, 1.6fr) minmax(90px, 1fr) auto;
  align-items: center;
  gap: 12px;
  padding: 9px 12px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: var(--color-bg);
  color: var(--color-text);
  font-size: 12px;
  text-align: left;
  cursor: pointer;
}

.bench-row.selected {
  border-color: var(--color-primary);
  box-shadow: 0 0 0 1px var(--color-primary);
}

.bench-row__name {
  font-weight: 600;
  font-size: 13px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.bench-row__plan,
.bench-row__when {
  color: var(--color-text-muted);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.status {
  font-size: 11px;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 999px;
  background: var(--color-surface);
  color: var(--color-text-muted);
}

.status--running,
.status--waiting {
  color: var(--status-running);
}

.status--done {
  color: var(--status-done);
}

.status--error {
  color: var(--status-error);
}

.bar {
  height: 6px;
  border-radius: 999px;
  background: var(--color-surface);
  overflow: hidden;
}

.bar__fill {
  display: block;
  height: 100%;
  background: var(--status-running);
}

.empty-state,
.muted {
  color: var(--color-text-muted);
  font-size: 13px;
  font-style: italic;
}

.empty-state {
  text-align: center;
  padding: 28px;
}

@media (max-width: 760px) {
  .bench-row {
    grid-template-columns: 1fr auto;
  }

  .bench-row__plan,
  .bench-row__when,
  .bar {
    grid-column: 1 / -1;
  }
}
</style>
