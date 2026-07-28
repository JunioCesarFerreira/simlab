<template>
  <div class="list-page">
    <div class="page-header">
      <h1 class="page-title">MILP Models</h1>
      <div class="header-actions">
        <span v-if="engineStatus" class="engine-chip" :class="engineChipClass">
          engine: {{ engineStatus.status }}
          <template v-if="engineStatus.available_backends.length">
            · {{ engineStatus.available_backends.join(", ") }}
          </template>
        </span>
        <button class="refresh-btn" :disabled="loading" @click="load">
          {{ loading ? "Loading…" : "Refresh" }}
        </button>
      </div>
    </div>

    <div v-if="error" class="error-banner">Failed to load: {{ error }}</div>

    <div v-if="engineStatus && engineStatus.gurobi_license" class="license-note">
      Gurobi: {{ engineStatus.gurobi_license }}
    </div>

    <div class="model-grid">
      <RouterLink
        v-for="m in models"
        :key="m.key"
        :to="`/models/${m.key}`"
        class="model-card"
      >
        <div class="model-title">{{ m.title }}</div>
        <div class="model-desc">{{ m.description }}</div>
        <div class="model-meta">
          <span class="badge">{{ m.problem_key }}</span>
          <span class="badge badge-key">{{ m.key }}</span>
        </div>
      </RouterLink>
    </div>

    <h2 class="section-title">Sweeps</h2>
    <div v-if="sweeps.length === 0 && !loading" class="empty-state">
      No sweeps yet. Open a model to launch a parameter sweep.
    </div>
    <table v-else class="sweep-table">
      <thead>
        <tr>
          <th>Name</th>
          <th>Model</th>
          <th>Status</th>
          <th>Progress</th>
          <th>Unique</th>
          <th>Created</th>
          <th>Experiment</th>
        </tr>
      </thead>
      <tbody>
        <tr
          v-for="s in sweeps"
          :key="s.id"
          class="sweep-row"
          @click="$router.push(`/milp-sweeps/${s.id}`)"
        >
          <td>{{ s.name }}</td>
          <td><span class="badge badge-key">{{ s.model_key }}</span></td>
          <td><span class="status-chip" :class="statusClass(s.status)">{{ s.status }}</span></td>
          <td>
            <div class="progress-cell">
              <div class="progress-bar">
                <div
                  class="progress-fill"
                  :style="{ width: progressPct(s) + '%' }"
                />
              </div>
              <span class="progress-label">
                {{ s.progress.done }}/{{ s.progress.total_combos }}
              </span>
            </div>
          </td>
          <td>{{ s.progress.unique_genotypes }}</td>
          <td>{{ formatDate(s.created_time) }}</td>
          <td>
            <RouterLink
              v-if="s.experiment_id"
              :to="`/experiments/${s.experiment_id}`"
              class="exp-link"
              @click.stop
            >
              open
            </RouterLink>
            <span v-else class="muted">—</span>
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from "vue";
import { getMilpEngineStatus, getMilpModels, listMilpSweeps } from "../api/milp";
import type { MilpEngineStatusDto, MilpModelInfoDto, MilpSweepInfoDto } from "../types/milp";

const models = ref<MilpModelInfoDto[]>([]);
const sweeps = ref<MilpSweepInfoDto[]>([]);
const engineStatus = ref<MilpEngineStatusDto | null>(null);
const loading = ref(false);
const error = ref<string | null>(null);

const engineChipClass = computed(() => ({
  "chip-online": engineStatus.value?.status === "online",
  "chip-unknown": engineStatus.value?.status !== "online",
}));

function progressPct(s: MilpSweepInfoDto): number {
  const total = s.progress.total_combos || 1;
  return Math.min(100, Math.round((s.progress.done / total) * 100));
}

function statusClass(status: string | null): string {
  switch (status) {
    case "Running": return "status-running";
    case "Done": return "status-done";
    case "Error": return "status-error";
    case "Cancelled": return "status-cancelled";
    default: return "status-waiting";
  }
}

function formatDate(iso: string | null): string {
  return iso ? new Date(iso).toLocaleString() : "—";
}

async function load() {
  loading.value = true;
  error.value = null;
  try {
    const [m, s, st] = await Promise.all([
      getMilpModels(),
      listMilpSweeps(),
      getMilpEngineStatus().catch(() => null),
    ]);
    models.value = m;
    sweeps.value = s;
    engineStatus.value = st;
  } catch (e: unknown) {
    error.value = e instanceof Error ? e.message : String(e);
  } finally {
    loading.value = false;
  }
}

let timer: ReturnType<typeof setInterval> | null = null;

onMounted(() => {
  load();
  timer = setInterval(async () => {
    // silent refresh of sweep progress while any sweep is active
    if (sweeps.value.some((s) => s.status === "Running" || s.status === "Waiting")) {
      try {
        sweeps.value = await listMilpSweeps();
      } catch {
        /* keep last known state */
      }
    }
  }, 5000);
});

onUnmounted(() => {
  if (timer) clearInterval(timer);
});
</script>

<style scoped>
.list-page { padding: 24px; max-width: 1100px; margin: 0 auto; }
.page-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
.page-title { font-size: 22px; font-weight: 800; color: var(--color-text); }
.header-actions { display: flex; gap: 10px; align-items: center; }

.refresh-btn {
  padding: 7px 14px; border-radius: var(--radius-md);
  border: 1px solid var(--color-border); background: var(--color-surface);
  color: var(--color-text); font-size: 13px; cursor: pointer;
}
.refresh-btn:hover { background: var(--color-surface-hover); }

.engine-chip {
  font-size: 12px; padding: 4px 10px; border-radius: 999px; font-weight: 600;
}
.chip-online { background: rgba(16, 185, 129, 0.12); color: #059669; }
.chip-unknown { background: rgba(148, 163, 184, 0.18); color: var(--color-text-muted); }

.license-note { font-size: 12px; color: var(--color-text-muted); margin-bottom: 14px; }

.error-banner {
  background: rgba(239, 68, 68, 0.08); color: #dc2626;
  padding: 10px 14px; border-radius: var(--radius-md); margin-bottom: 14px; font-size: 13px;
}

.model-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 14px; margin-bottom: 28px; }
.model-card {
  display: block; padding: 16px; border: 1px solid var(--color-border);
  border-radius: var(--radius-md); background: var(--color-surface);
  transition: border-color 0.12s, box-shadow 0.12s;
}
.model-card:hover { border-color: var(--color-primary); box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
.model-title { font-size: 15px; font-weight: 700; color: var(--color-text); margin-bottom: 6px; }
.model-desc { font-size: 13px; color: var(--color-text-muted); line-height: 1.5; margin-bottom: 10px; }
.model-meta { display: flex; gap: 6px; }

.badge {
  font-size: 11px; padding: 2px 8px; border-radius: 999px;
  background: var(--color-primary-light); color: var(--color-primary); font-weight: 600;
}
.badge-key { background: rgba(148, 163, 184, 0.15); color: var(--color-text-muted); }

.section-title { font-size: 16px; font-weight: 700; color: var(--color-text); margin: 8px 0 12px; }
.empty-state { color: var(--color-text-muted); font-size: 13px; padding: 18px 0; }

.sweep-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.sweep-table th {
  text-align: left; padding: 8px 10px; color: var(--color-text-muted);
  font-weight: 600; border-bottom: 1px solid var(--color-border); font-size: 12px;
}
.sweep-table td { padding: 9px 10px; border-bottom: 1px solid var(--color-border); color: var(--color-text); }
.sweep-row { cursor: pointer; }
.sweep-row:hover { background: var(--color-surface-hover); }

.status-chip { font-size: 11px; padding: 2px 8px; border-radius: 999px; font-weight: 600; }
.status-running { background: rgba(59, 130, 246, 0.12); color: #2563eb; }
.status-done { background: rgba(16, 185, 129, 0.12); color: #059669; }
.status-error { background: rgba(239, 68, 68, 0.12); color: #dc2626; }
.status-cancelled { background: rgba(148, 163, 184, 0.18); color: var(--color-text-muted); }
.status-waiting { background: rgba(245, 158, 11, 0.12); color: #d97706; }

.progress-cell { display: flex; align-items: center; gap: 8px; }
.progress-bar { width: 90px; height: 6px; border-radius: 3px; background: rgba(148, 163, 184, 0.25); overflow: hidden; }
.progress-fill { height: 100%; background: var(--color-primary); border-radius: 3px; transition: width 0.4s; }
.progress-label { font-size: 11px; color: var(--color-text-muted); }

.exp-link { color: var(--color-primary); font-weight: 600; }
.muted { color: var(--color-text-muted); }
</style>
