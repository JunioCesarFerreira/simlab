<template>
  <div class="detail-page">
    <div v-if="loadError" class="error-banner">{{ loadError }}</div>

    <template v-if="model">
      <div class="page-header">
        <div>
          <h1 class="page-title">{{ model.title }}</h1>
          <div class="model-meta">
            <span class="badge">{{ model.problem_key }}</span>
            <span class="badge badge-key">{{ model.key }}</span>
          </div>
        </div>
        <RouterLink to="/models" class="back-link">← Models</RouterLink>
      </div>

      <p class="model-desc">{{ model.description }}</p>

      <details class="formulation-box">
        <summary>Mathematical formulation</summary>
        <pre class="formulation">{{ model.formulation }}</pre>
      </details>

      <!-- ─────────────── Sweep launch form ─────────────── -->
      <h2 class="section-title">New parameter sweep</h2>

      <div class="form-grid">
        <div class="field span-2">
          <label class="field-label">Sweep name <span class="required">*</span></label>
          <input v-model="form.name" class="field-input" placeholder="e.g. P2 sweep — instance 40 candidates" />
        </div>

        <div class="field span-2">
          <label class="field-label">Problem ({{ model.problem_key }}) <span class="required">*</span></label>
          <select v-model="form.problemId" class="field-input" @change="onProblemSelected">
            <option value="" disabled>Select a problem draft…</option>
            <option v-for="p in problems" :key="p.id" :value="p.id">{{ p.name }}</option>
          </select>
          <span v-if="problemError" class="field-error">{{ problemError }}</span>
          <span v-else-if="problemSummary" class="field-hint">{{ problemSummary }}</span>
        </div>
      </div>

      <!-- Parameters -->
      <h3 class="subsection-title">Model parameters</h3>
      <table class="param-table">
        <thead>
          <tr><th>Parameter</th><th>Mode</th><th>Value(s)</th><th></th></tr>
        </thead>
        <tbody>
          <tr v-for="row in paramRows" :key="row.name">
            <td>
              <span class="param-name">{{ row.name }}</span>
              <div class="param-desc">{{ row.description }}</div>
            </td>
            <td>
              <select v-model="row.mode" class="field-input compact" :disabled="!row.sweepable">
                <option value="fixed">fixed</option>
                <option value="sweep">sweep</option>
              </select>
            </td>
            <td>
              <input
                v-if="row.mode === 'fixed'"
                v-model="row.fixedValue"
                type="number"
                step="any"
                class="field-input compact"
              />
              <input
                v-else
                v-model="row.sweepValues"
                class="field-input"
                placeholder="comma-separated, e.g. 10, 110, 310, 610, 1010"
              />
            </td>
            <td class="count-cell">
              <span v-if="row.mode === 'sweep'" class="value-count">{{ parseValues(row.sweepValues).length }} values</span>
            </td>
          </tr>
        </tbody>
      </table>
      <div class="combo-counter" :class="{ 'combo-warn': comboCount > 1000 }">
        {{ comboCount }} combination{{ comboCount === 1 ? "" : "s" }}
      </div>

      <!-- Solver -->
      <h3 class="subsection-title">Solver</h3>
      <div class="form-grid">
        <div class="field">
          <label class="field-label">Backend</label>
          <select v-model="form.solver.backend" class="field-input">
            <option value="gurobi">Gurobi</option>
            <option value="highs">HiGHS (open source)</option>
          </select>
          <span v-if="engineStatus" class="field-hint">
            available: {{ engineStatus.available_backends.join(", ") || "unknown" }}
          </span>
        </div>
        <div class="field">
          <label class="field-label">Time limit (s / solve)</label>
          <input v-model.number="form.solver.time_limit_s" type="number" min="1" class="field-input" />
        </div>
        <div class="field">
          <label class="field-label">MIP gap</label>
          <input v-model.number="form.solver.mip_gap" type="number" step="0.001" min="0" class="field-input" />
        </div>
        <div class="field field-check">
          <label class="check-label">
            <input v-model="form.solver.allow_fallback" type="checkbox" />
            Fall back to another backend if unavailable
          </label>
        </div>
      </div>

      <!-- Batch experiment options -->
      <h3 class="subsection-title">Batch experiment (simulation of the topologies)</h3>
      <div class="form-grid">
        <div class="field">
          <label class="field-label">Simulation duration (s)</label>
          <input v-model.number="form.simDuration" type="number" min="1" class="field-input" />
        </div>
        <div class="field">
          <label class="field-label">Random seeds (comma-separated)</label>
          <input v-model="form.randomSeeds" class="field-input" />
        </div>
        <div class="field field-check">
          <label class="check-label"><input v-model="form.macCsma" type="checkbox" /> CSMA/CA</label>
          <label class="check-label"><input v-model="form.macTsch" type="checkbox" /> TSCH</label>
        </div>
      </div>

      <div class="field">
        <label class="field-label">Source repositories <span class="required">*</span></label>
        <div v-for="(opt, i) in form.sourceOptions" :key="i" class="source-row">
          <input v-model="opt.protocol" class="field-input compact" placeholder="protocol (csma/tsch)" />
          <select v-model="opt.repoId" class="field-input">
            <option value="" disabled>Select repository…</option>
            <option v-for="r in repositories" :key="r.id" :value="r.id">{{ r.name }}</option>
          </select>
          <button class="icon-btn" title="Remove" @click="form.sourceOptions.splice(i, 1)">✕</button>
        </div>
        <button class="add-btn" @click="form.sourceOptions.push({ protocol: '', repoId: '' })">+ Add mapping</button>
      </div>

      <div class="field">
        <label class="field-label">Objectives <span class="required">*</span></label>
        <div v-for="(obj, i) in form.objectives" :key="i" class="source-row">
          <input v-model="obj.metric_name" class="field-input" placeholder="metric name" />
          <select v-model="obj.goal" class="field-input compact">
            <option value="min">min</option>
            <option value="max">max</option>
          </select>
          <button class="icon-btn" title="Remove" @click="form.objectives.splice(i, 1)">✕</button>
        </div>
        <button class="add-btn" @click="form.objectives.push({ metric_name: '', goal: 'min' })">+ Add objective</button>
      </div>

      <div v-if="submitError" class="error-banner">{{ submitError }}</div>

      <div class="submit-row">
        <button class="primary-btn" :disabled="submitting || !canSubmit" @click="submit">
          {{ submitting ? "Launching…" : `Launch sweep (${comboCount} solves)` }}
        </button>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref } from "vue";
import { useRouter } from "vue-router";
import { createMilpSweep, getMilpEngineStatus, getMilpModel } from "../api/milp";
import { getProblem, listProblems, type ProblemInfoDto } from "../api/problems";
import { getAllRepositories } from "../api/repositories";
import { exportProblem } from "../services/exportProblemJson";
import type { MilpEngineStatusDto, MilpModelDto } from "../types/milp";
import type { SourceRepositoryDto, ObjectiveItem } from "../types/simlab";

const props = defineProps<{ modelKey: string }>();
const router = useRouter();

const model = ref<MilpModelDto | null>(null);
const engineStatus = ref<MilpEngineStatusDto | null>(null);
const problems = ref<ProblemInfoDto[]>([]);
const repositories = ref<SourceRepositoryDto[]>([]);
const loadError = ref<string | null>(null);
const problemError = ref<string | null>(null);
const problemSummary = ref<string | null>(null);
const submitError = ref<string | null>(null);
const submitting = ref(false);

// exported problem JSON, produced client-side from the selected draft
const exportedProblem = ref<Record<string, unknown> | null>(null);

interface ParamRow {
  name: string;
  description: string;
  sweepable: boolean;
  mode: "fixed" | "sweep";
  fixedValue: number;
  sweepValues: string;
}
const paramRows = ref<ParamRow[]>([]);

const form = reactive({
  name: "",
  problemId: "",
  solver: { backend: "gurobi", time_limit_s: 300, mip_gap: 0.01, allow_fallback: true },
  simDuration: 180,
  randomSeeds: "336157, 667370, 35239",
  macCsma: true,
  macTsch: false,
  sourceOptions: [{ protocol: "csma", repoId: "" }] as { protocol: string; repoId: string }[],
  objectives: [
    { metric_name: "latency", goal: "min" },
    { metric_name: "energy", goal: "min" },
    { metric_name: "throughput", goal: "max" },
  ] as ObjectiveItem[],
});

// Same defaults the LaunchWizard uses for Cooja CSV -> metric conversion
const DEFAULT_DATA_CONVERSION = {
  node_col: "node",
  time_col: "root_time_now",
  metrics: [
    { name: "cpu_energy_mj", kind: "sum_all", column: "cpu_energy_mj" },
    { name: "lpm_energy_mj", kind: "sum_all", column: "lpm_energy_mj" },
    { name: "radio_tx_energy_mj", kind: "sum_all", column: "radio_tx_energy_mj" },
    { name: "radio_rx_energy_mj", kind: "sum_all", column: "radio_rx_energy_mj" },
    { name: "total_sent", kind: "sum_last_minus_first", column: "total_sent" },
    { name: "total_received", kind: "sum_last_minus_first", column: "total_received" },
    { name: "server_sent", kind: "sum_last_minus_first", column: "server_received" },
    { name: "bytes_tx", kind: "sum_last_minus_first", column: "bytes_tx" },
    { name: "bytes_rx", kind: "sum_last_minus_first", column: "bytes_rx" },
    { name: "server_bytes_rx", kind: "sum_last_minus_first", column: "server_bytes_rx" },
    { name: "r2n_latency", kind: "mean", column: "r2n_latency" },
    { name: "n2r_latency", kind: "mean", column: "n2r_latency" },
    { name: "hops", kind: "mean", column: "hops" },
    { name: "rtt_latency", kind: "mean", column: "rtt_latency" },
    { name: "latency", kind: "mean", column: "rtt_latency" },
    { name: "energy", kind: "sum_all", column: "total_energy_mj" },
    { name: "throughput", kind: "sum_last_minus_first", column: "server_received" },
  ],
};

function parseValues(text: string): number[] {
  return text
    .split(/[\s,;]+/)
    .filter((s) => s.length > 0)
    .map(Number)
    .filter((v) => Number.isFinite(v));
}

const comboCount = computed(() =>
  paramRows.value
    .filter((r) => r.mode === "sweep")
    .reduce((acc, r) => acc * Math.max(parseValues(r.sweepValues).length, 0), 1),
);

const canSubmit = computed(
  () =>
    form.name.trim().length > 0 &&
    exportedProblem.value !== null &&
    !problemError.value &&
    comboCount.value >= 1 &&
    (form.macCsma || form.macTsch) &&
    form.sourceOptions.length > 0 &&
    form.sourceOptions.every((o) => o.protocol.trim() && o.repoId) &&
    form.objectives.length > 0 &&
    form.objectives.every((o) => o.metric_name.trim().length > 0),
);

async function onProblemSelected() {
  problemError.value = null;
  problemSummary.value = null;
  exportedProblem.value = null;
  if (!form.problemId || !model.value) return;
  try {
    const doc = await getProblem(form.problemId);
    const exported = exportProblem(doc.draft) as { problem: Record<string, unknown> };
    const problem = exported.problem;
    if (problem.name !== model.value.problem_key) {
      problemError.value = `This draft is a '${String(problem.name)}' problem; the model needs '${model.value.problem_key}'.`;
      return;
    }
    exportedProblem.value = problem;
    const candidates = (problem.candidates as unknown[] | undefined)?.length ?? 0;
    const mobiles = (problem.mobile_nodes as unknown[] | undefined)?.length ?? 0;
    problemSummary.value = `${candidates} candidates · ${mobiles} mobile node${mobiles === 1 ? "" : "s"}`;
  } catch (e: unknown) {
    problemError.value = e instanceof Error ? e.message : String(e);
  }
}

async function submit() {
  if (!model.value || !exportedProblem.value) return;
  submitting.value = true;
  submitError.value = null;

  const parameter_grid: Record<string, number[]> = {};
  const fixed_parameters: Record<string, number> = {};
  for (const row of paramRows.value) {
    if (row.mode === "sweep") parameter_grid[row.name] = parseValues(row.sweepValues);
    else fixed_parameters[row.name] = Number(row.fixedValue);
  }

  const mac_protocols = [
    ...(form.macCsma ? [0] : []),
    ...(form.macTsch ? [1] : []),
  ];
  const source_repository_options: Record<string, string> = {};
  for (const { protocol, repoId } of form.sourceOptions) {
    if (protocol && repoId) source_repository_options[protocol] = repoId;
  }

  try {
    const sweepId = await createMilpSweep({
      name: form.name.trim(),
      model_key: model.value.key,
      problem: exportedProblem.value,
      problem_id: form.problemId || null,
      parameter_grid,
      fixed_parameters,
      solver: form.solver,
      batch_options: {
        simulation: {
          duration: form.simDuration,
          random_seeds: parseValues(form.randomSeeds).map((v) => Math.trunc(v)),
        },
        objectives: form.objectives,
        mac_protocols,
        source_repository_options,
        data_conversion_config: DEFAULT_DATA_CONVERSION,
      },
    });
    router.push(`/milp-sweeps/${sweepId}`);
  } catch (e: unknown) {
    const detail =
      (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
    submitError.value = detail ?? (e instanceof Error ? e.message : String(e));
  } finally {
    submitting.value = false;
  }
}

onMounted(async () => {
  try {
    model.value = await getMilpModel(props.modelKey);
    form.name = `${model.value.title} sweep`;
    form.solver.time_limit_s = model.value.solver_defaults.time_limit_s ?? 300;
    form.solver.mip_gap = model.value.solver_defaults.mip_gap ?? 0.01;
    paramRows.value = model.value.parameters.map((p) => ({
      name: p.name,
      description: p.description,
      sweepable: p.sweepable,
      mode: "fixed",
      fixedValue: p.default,
      sweepValues: String(p.default),
    }));
  } catch (e: unknown) {
    loadError.value = e instanceof Error ? e.message : String(e);
    return;
  }
  engineStatus.value = await getMilpEngineStatus().catch(() => null);
  problems.value = await listProblems().catch(() => []);
  repositories.value = await getAllRepositories().catch(() => []);
});
</script>

<style scoped>
.detail-page { padding: 24px; max-width: 980px; margin: 0 auto; }
.page-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 10px; }
.page-title { font-size: 22px; font-weight: 800; color: var(--color-text); margin-bottom: 6px; }
.back-link { color: var(--color-text-muted); font-size: 13px; }
.back-link:hover { color: var(--color-primary); }

.model-meta { display: flex; gap: 6px; }
.badge {
  font-size: 11px; padding: 2px 8px; border-radius: 999px;
  background: var(--color-primary-light); color: var(--color-primary); font-weight: 600;
}
.badge-key { background: rgba(148, 163, 184, 0.15); color: var(--color-text-muted); }
.model-desc { font-size: 13px; color: var(--color-text-muted); line-height: 1.6; margin-bottom: 14px; }

.formulation-box { margin-bottom: 22px; border: 1px solid var(--color-border); border-radius: var(--radius-md); }
.formulation-box summary {
  padding: 9px 14px; font-size: 13px; font-weight: 600;
  color: var(--color-text); cursor: pointer;
}
.formulation {
  margin: 0; padding: 12px 14px; font-size: 12px; line-height: 1.6;
  overflow-x: auto; color: var(--color-text-muted);
  border-top: 1px solid var(--color-border); white-space: pre-wrap;
}

.section-title { font-size: 16px; font-weight: 700; color: var(--color-text); margin: 20px 0 12px; }
.subsection-title { font-size: 14px; font-weight: 700; color: var(--color-text); margin: 20px 0 10px; }

.form-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px 16px; }
.span-2 { grid-column: span 2; }

.field { display: flex; flex-direction: column; gap: 5px; margin-bottom: 10px; }
.field-label { font-size: 12px; font-weight: 600; color: var(--color-text-muted); }
.required { color: #dc2626; }
.field-input {
  padding: 7px 10px; border: 1px solid var(--color-border); border-radius: var(--radius-md);
  background: var(--color-surface); color: var(--color-text); font-size: 13px; width: 100%;
}
.field-input.compact { width: 130px; }
.field-hint { font-size: 11px; color: var(--color-text-muted); }
.field-error { font-size: 11px; color: #dc2626; }
.field-check { justify-content: flex-end; flex-direction: row; gap: 16px; align-items: center; }
.check-label { display: flex; align-items: center; gap: 6px; font-size: 13px; color: var(--color-text); }

.param-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.param-table th {
  text-align: left; padding: 6px 8px; color: var(--color-text-muted);
  font-size: 12px; font-weight: 600; border-bottom: 1px solid var(--color-border);
}
.param-table td { padding: 8px; border-bottom: 1px solid var(--color-border); vertical-align: top; }
.param-name { font-weight: 600; color: var(--color-text); font-family: monospace; }
.param-desc { font-size: 11px; color: var(--color-text-muted); margin-top: 2px; }
.count-cell { width: 80px; }
.value-count { font-size: 11px; color: var(--color-text-muted); }

.combo-counter { margin: 10px 0 4px; font-size: 13px; font-weight: 600; color: var(--color-text); }
.combo-warn { color: #d97706; }

.source-row { display: flex; gap: 8px; margin-bottom: 6px; align-items: center; }
.icon-btn {
  border: none; background: transparent; color: var(--color-text-muted);
  cursor: pointer; font-size: 13px; padding: 4px;
}
.icon-btn:hover { color: #dc2626; }
.add-btn {
  align-self: flex-start; border: 1px dashed var(--color-border); background: transparent;
  color: var(--color-text-muted); font-size: 12px; padding: 5px 10px;
  border-radius: var(--radius-md); cursor: pointer;
}
.add-btn:hover { color: var(--color-primary); border-color: var(--color-primary); }

.error-banner {
  background: rgba(239, 68, 68, 0.08); color: #dc2626;
  padding: 10px 14px; border-radius: var(--radius-md); margin: 12px 0; font-size: 13px;
}

.submit-row { margin-top: 18px; padding-bottom: 30px; }
.primary-btn {
  padding: 9px 18px; border-radius: var(--radius-md); border: none;
  background: var(--color-primary); color: #fff; font-size: 14px; font-weight: 600; cursor: pointer;
}
.primary-btn:disabled { opacity: 0.5; cursor: not-allowed; }
</style>
