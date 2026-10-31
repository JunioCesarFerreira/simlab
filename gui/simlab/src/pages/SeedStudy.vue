<template>
  <div class="study">
    <div class="page-header">
      <div>
        <h1 class="page-title">Seed Study</h1>
        <p class="subtitle">
          Run the same solutions under a list of seeds, see how every metric is distributed, and
          measure how many seeds it takes to estimate a value and to rank solutions.
        </p>
      </div>
      <div class="header-actions">
        <button class="refresh-btn" :disabled="loading" @click="load()">
          {{ loading ? "Loading…" : "Refresh" }}
        </button>
        <button class="primary-btn" @click="showForm = !showForm">
          {{ showForm ? "Close" : "New study" }}
        </button>
      </div>
    </div>

    <p v-if="error" class="error-banner">{{ error }}</p>
    <p v-if="notice" class="notice-banner">{{ notice }}</p>

    <section v-if="showForm" class="card">
      <h2 class="card-title">New study</h2>
      <SeedStudyForm :pool="pool" @created="onCreated" />
    </section>

    <section class="card">
      <h2 class="card-title">Studies</h2>
      <div v-if="!studies.length && !loading" class="empty-state">
        No study yet. Start one to measure how many seeds an instance needs.
      </div>
      <ul v-else class="bench-list">
        <li v-for="s in studies" :key="s.id">
          <button
            type="button"
            :class="['bench-row', { selected: s.id === selectedId }]"
            :aria-pressed="s.id === selectedId"
            @click="select(s.id)"
          >
            <span class="bench-row__name">{{ s.name }}</span>
            <span :class="['status', `status--${s.status.toLowerCase()}`]">{{ s.status }}</span>
            <span class="bench-row__plan">
              {{ s.chromosomes }} × {{ s.seeds }} · {{ s.simulations }} sims
            </span>
            <span
              class="bar"
              role="progressbar"
              :aria-valuenow="s.finished ?? 0"
              aria-valuemin="0"
              :aria-valuemax="s.simulations"
              :aria-label="`${s.finished ?? 0} of ${s.simulations} simulations finished`"
            >
              <span class="bar__fill" :style="{ width: `${progress(s) * 100}%` }" />
            </span>
            <span class="bench-row__when">{{ when(s) }}</span>
          </button>
        </li>
      </ul>
    </section>

    <section v-if="detail" class="card">
      <SeedStudyResult :detail="detail" @targets="onTargets" />
    </section>
    <p v-else-if="selectedId && detailLoading" class="muted">Loading study…</p>
  </div>
</template>

<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from "vue";
import { useRoute, useRouter } from "vue-router";
import SeedStudyForm from "../components/seed-study/SeedStudyForm.vue";
import SeedStudyResult from "../components/seed-study/SeedStudyResult.vue";
import { getOverview } from "../api/infra";
import { getStudy, listStudies } from "../api/seedStudy";
import { progress } from "../services/seedStudy";
import { schedulablePool } from "../services/usl";
import type { StudyCreated, StudyDetail, StudySummary, StudyTargets } from "../types/seedStudy";

/** How often a running study is re-read. */
const POLL_MS = 10_000;

const route = useRoute();
const router = useRouter();

const studies = ref<StudySummary[]>([]);
const detail = ref<StudyDetail | null>(null);
const selectedId = ref<string | null>((route.query.id as string) || null);
const targets = ref<StudyTargets | null>(null);
const pool = ref(0);
const loading = ref(false);
const detailLoading = ref(false);
const showForm = ref(false);
const error = ref("");
const notice = ref("");

let poll: ReturnType<typeof setInterval> | null = null;
let inFlight: AbortController | null = null;

function describe(e: unknown): string {
  const message = (e as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail;
  if (typeof message === "string") return message;
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
      listStudies(controller.signal),
      getOverview(controller.signal),
    ]);
    studies.value = list;
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
    detail.value = await getStudy(selectedId.value, targets.value, signal);
  } finally {
    detailLoading.value = false;
  }
}

async function select(id: string) {
  selectedId.value = id;
  detail.value = null;
  targets.value = null;
  router.replace({ query: { ...route.query, id } });
  try {
    await loadDetail();
  } catch (e) {
    error.value = describe(e);
  }
}

/** New targets recompute the analysis from the same runs. */
async function onTargets(next: StudyTargets) {
  targets.value = next;
  try {
    await loadDetail();
    error.value = "";
  } catch (e) {
    error.value = describe(e);
  }
}

async function onCreated(created: StudyCreated) {
  showForm.value = false;
  notice.value =
    `Study queued: ${created.chromosomes} solution(s) × ${created.seeds} seeds, ` +
    `${created.simulations} simulation(s). The analysis fills in as they finish.`;
  selectedId.value = created.id;
  targets.value = null;
  await load();
}

function when(s: StudySummary): string {
  const stamp = s.start_time || s.created_time;
  return stamp ? new Date(stamp).toLocaleString() : "";
}

onMounted(async () => {
  await load();
  // Re-read only while something is moving: a finished study does not change.
  poll = setInterval(() => {
    const active = studies.value.some((s) => running(s.status));
    if (active && !showForm.value) load(false);
  }, POLL_MS);
});

onBeforeUnmount(() => {
  if (poll) clearInterval(poll);
  inFlight?.abort();
});
</script>

<style scoped>
.study {
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
  grid-template-columns: minmax(140px, 1.4fr) auto minmax(180px, 1.6fr) minmax(90px, 1fr) auto;
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
