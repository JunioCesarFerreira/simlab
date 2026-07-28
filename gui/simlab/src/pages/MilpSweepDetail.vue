<template>
  <div class="detail-page">
    <div v-if="error" class="error-banner">{{ error }}</div>

    <template v-if="sweep">
      <div class="page-header">
        <div>
          <h1 class="page-title">{{ sweep.name }}</h1>
          <div class="meta-row">
            <span class="badge badge-key">{{ sweep.model_key }}</span>
            <span class="status-chip" :class="statusClass(sweep.status)">{{ sweep.status }}</span>
            <span v-if="sweep.cancel_requested && sweep.status === 'Running'" class="cancel-note">
              cancellation requested…
            </span>
          </div>
        </div>
        <div class="header-actions">
          <button
            v-if="sweep.status === 'Running' || sweep.status === 'Waiting'"
            class="danger-btn"
            :disabled="cancelling || sweep.cancel_requested"
            @click="cancel"
          >
            {{ cancelling ? "Cancelling…" : "Cancel sweep" }}
          </button>
          <RouterLink to="/models" class="back-link">← Models</RouterLink>
        </div>
      </div>

      <div v-if="sweep.system_message" class="system-message">{{ sweep.system_message }}</div>

      <!-- progress -->
      <div class="progress-section">
        <div class="progress-bar">
          <div class="progress-fill" :style="{ width: progressPct + '%' }" />
        </div>
        <div class="stats-row">
          <div class="stat"><span class="stat-value">{{ sweep.progress.done }}/{{ sweep.progress.total_combos }}</span><span class="stat-label">combinations</span></div>
          <div class="stat"><span class="stat-value">{{ sweep.progress.solved }}</span><span class="stat-label">solved</span></div>
          <div class="stat"><span class="stat-value">{{ sweep.progress.infeasible }}</span><span class="stat-label">infeasible</span></div>
          <div class="stat"><span class="stat-value">{{ sweep.progress.unique_genotypes }}</span><span class="stat-label">unique topologies</span></div>
          <div class="stat">
            <span class="stat-value">
              <RouterLink v-if="sweep.experiment_id" :to="`/experiments/${sweep.experiment_id}`" class="exp-link">open →</RouterLink>
              <span v-else class="muted">—</span>
            </span>
            <span class="stat-label">batch experiment</span>
          </div>
        </div>
      </div>

      <!-- configuration summary -->
      <h2 class="section-title">Configuration</h2>
      <div class="config-grid">
        <div class="config-box">
          <div class="config-title">Swept parameters</div>
          <div v-for="(values, name) in sweep.parameter_grid" :key="name" class="config-line">
            <span class="param-name">{{ name }}</span> = [{{ values.join(", ") }}]
          </div>
        </div>
        <div class="config-box">
          <div class="config-title">Fixed parameters</div>
          <div v-for="(value, name) in sweep.fixed_parameters" :key="name" class="config-line">
            <span class="param-name">{{ name }}</span> = {{ value }}
          </div>
          <div class="config-line muted">
            solver: {{ sweep.solver.backend }} · {{ sweep.solver.time_limit_s }}s · gap {{ sweep.solver.mip_gap }}
          </div>
        </div>
      </div>

      <!-- solutions -->
      <h2 class="section-title">Solutions ({{ sweep.solutions.length }})</h2>
      <div v-if="sweep.solutions.length === 0" class="empty-state">No combinations processed yet.</div>
      <table v-else class="sol-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Parameters</th>
            <th>Status</th>
            <th>Genotype</th>
            <th>Installed</th>
            <th>Objective</th>
            <th>Gap</th>
            <th>Time (s)</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="r in sweep.solutions" :key="r.index" :class="{ 'row-dup': r.is_duplicate }">
            <td>{{ r.index }}</td>
            <td class="params-cell">{{ formatParams(r.params) }}</td>
            <td>
              <span class="status-chip" :class="solveStatusClass(r.status)">{{ r.status }}</span>
              <span v-if="r.is_duplicate" class="dup-badge">dup</span>
            </td>
            <td class="genotype-cell">{{ r.genotype ?? "—" }}</td>
            <td>{{ r.n_installed ?? "—" }}</td>
            <td>{{ r.obj_value != null ? r.obj_value.toExponential(3) : "—" }}</td>
            <td>{{ r.mip_gap != null ? (r.mip_gap * 100).toFixed(2) + "%" : "—" }}</td>
            <td>{{ r.runtime_s.toFixed(1) }}</td>
          </tr>
        </tbody>
      </table>
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from "vue";
import { cancelMilpSweep, getMilpSweep } from "../api/milp";
import type { MilpSweepDto } from "../types/milp";

const props = defineProps<{ id: string }>();

const sweep = ref<MilpSweepDto | null>(null);
const error = ref<string | null>(null);
const cancelling = ref(false);

const progressPct = computed(() => {
  if (!sweep.value) return 0;
  const total = sweep.value.progress.total_combos || 1;
  return Math.min(100, Math.round((sweep.value.progress.done / total) * 100));
});

function statusClass(status: string | null): string {
  switch (status) {
    case "Running": return "status-running";
    case "Done": return "status-done";
    case "Error": return "status-error";
    case "Cancelled": return "status-cancelled";
    default: return "status-waiting";
  }
}

function solveStatusClass(status: string): string {
  switch (status) {
    case "OPTIMAL": return "status-done";
    case "TIME_LIMIT": return "status-waiting";
    case "INFEASIBLE": return "status-cancelled";
    default: return "status-error";
  }
}

function formatParams(params: Record<string, number>): string {
  return Object.entries(params)
    .map(([k, v]) => `${k}=${v}`)
    .join("  ");
}

async function load() {
  try {
    sweep.value = await getMilpSweep(props.id);
    error.value = null;
  } catch (e: unknown) {
    error.value = e instanceof Error ? e.message : String(e);
  }
}

async function cancel() {
  cancelling.value = true;
  try {
    await cancelMilpSweep(props.id);
    await load();
  } catch (e: unknown) {
    const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
    error.value = detail ?? (e instanceof Error ? e.message : String(e));
  } finally {
    cancelling.value = false;
  }
}

let timer: ReturnType<typeof setInterval> | null = null;

onMounted(() => {
  load();
  timer = setInterval(() => {
    const st = sweep.value?.status;
    if (st === "Running" || st === "Waiting" || st == null) load();
  }, 4000);
});

onUnmounted(() => {
  if (timer) clearInterval(timer);
});
</script>

<style scoped>
.detail-page { padding: 24px; max-width: 1100px; margin: 0 auto; }
.page-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px; }
.page-title { font-size: 22px; font-weight: 800; color: var(--color-text); margin-bottom: 6px; }
.header-actions { display: flex; gap: 12px; align-items: center; }
.back-link { color: var(--color-text-muted); font-size: 13px; }
.back-link:hover { color: var(--color-primary); }

.meta-row { display: flex; gap: 8px; align-items: center; }
.badge-key {
  font-size: 11px; padding: 2px 8px; border-radius: 999px;
  background: rgba(148, 163, 184, 0.15); color: var(--color-text-muted); font-weight: 600;
}
.cancel-note { font-size: 12px; color: #d97706; }

.status-chip { font-size: 11px; padding: 2px 8px; border-radius: 999px; font-weight: 600; }
.status-running { background: rgba(59, 130, 246, 0.12); color: #2563eb; }
.status-done { background: rgba(16, 185, 129, 0.12); color: #059669; }
.status-error { background: rgba(239, 68, 68, 0.12); color: #dc2626; }
.status-cancelled { background: rgba(148, 163, 184, 0.18); color: var(--color-text-muted); }
.status-waiting { background: rgba(245, 158, 11, 0.12); color: #d97706; }

.danger-btn {
  padding: 7px 14px; border-radius: var(--radius-md); border: 1px solid rgba(239, 68, 68, 0.4);
  background: transparent; color: #dc2626; font-size: 13px; cursor: pointer;
}
.danger-btn:hover:not(:disabled) { background: rgba(239, 68, 68, 0.08); }
.danger-btn:disabled { opacity: 0.5; cursor: not-allowed; }

.system-message {
  background: rgba(245, 158, 11, 0.08); color: #b45309;
  padding: 10px 14px; border-radius: var(--radius-md); margin-bottom: 14px; font-size: 13px;
}
.error-banner {
  background: rgba(239, 68, 68, 0.08); color: #dc2626;
  padding: 10px 14px; border-radius: var(--radius-md); margin-bottom: 14px; font-size: 13px;
}

.progress-section { margin-bottom: 20px; }
.progress-bar {
  height: 8px; border-radius: 4px; background: rgba(148, 163, 184, 0.25);
  overflow: hidden; margin-bottom: 12px;
}
.progress-fill { height: 100%; background: var(--color-primary); border-radius: 4px; transition: width 0.4s; }

.stats-row { display: flex; gap: 28px; flex-wrap: wrap; }
.stat { display: flex; flex-direction: column; gap: 2px; }
.stat-value { font-size: 16px; font-weight: 700; color: var(--color-text); }
.stat-label { font-size: 11px; color: var(--color-text-muted); }
.exp-link { color: var(--color-primary); font-weight: 700; font-size: 14px; }
.muted { color: var(--color-text-muted); }

.section-title { font-size: 16px; font-weight: 700; color: var(--color-text); margin: 20px 0 12px; }
.empty-state { color: var(--color-text-muted); font-size: 13px; padding: 12px 0; }

.config-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; }
.config-box { border: 1px solid var(--color-border); border-radius: var(--radius-md); padding: 12px 14px; }
.config-title { font-size: 12px; font-weight: 700; color: var(--color-text-muted); margin-bottom: 8px; }
.config-line { font-size: 13px; color: var(--color-text); margin-bottom: 4px; }
.param-name { font-family: monospace; font-weight: 600; }

.sol-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.sol-table th {
  text-align: left; padding: 7px 9px; color: var(--color-text-muted);
  font-weight: 600; border-bottom: 1px solid var(--color-border); font-size: 12px;
}
.sol-table td { padding: 7px 9px; border-bottom: 1px solid var(--color-border); color: var(--color-text); }
.row-dup { opacity: 0.55; }
.params-cell { font-family: monospace; font-size: 12px; }
.genotype-cell {
  font-family: monospace; font-size: 12px; max-width: 220px;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.dup-badge {
  margin-left: 6px; font-size: 10px; padding: 1px 6px; border-radius: 999px;
  background: rgba(148, 163, 184, 0.18); color: var(--color-text-muted); font-weight: 700;
}
</style>
