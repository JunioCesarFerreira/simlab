<template>
  <Teleport to="body">
    <div class="backdrop" @click.self="$emit('close')" />
    <div class="panel">
      <!-- Header -->
      <div class="panel-header">
        <div class="panel-title">
          <span class="label">Indivíduo</span>
          <span class="hash mono">{{ individual.individual_id }}</span>
        </div>
        <button class="close-btn" @click="$emit('close')">✕</button>
      </div>

      <div class="panel-body">
        <!-- Objetivos -->
        <section class="section">
          <div class="section-title">Objetivos</div>
          <div class="objectives-grid">
            <div
              v-for="(val, i) in individual.objectives"
              :key="i"
              class="obj-card"
            >
              <div class="obj-name">{{ objectiveNames[i] ?? `obj${i}` }}</div>
              <div class="obj-val">{{ val.toFixed(6) }}</div>
            </div>
          </div>
        </section>

        <!-- Topologia -->
        <section v-if="individual.topology_picture_id" class="section">
          <div class="section-title">Topologia</div>
          <div class="topology-wrap">
            <div v-if="loadingTopology" class="topology-placeholder">Carregando…</div>
            <img
              v-else-if="topologyBlobUrl"
              :src="topologyBlobUrl"
              class="topology-img"
              alt="Topologia da rede"
            />
            <div v-else class="topology-placeholder error">Não disponível</div>
          </div>
        </section>

        <!-- Cromossomo -->
        <section class="section">
          <div class="section-title">Cromossomo</div>
          <div class="kv-table">
            <div
              v-for="(val, key) in individual.chromosome"
              :key="key"
              class="kv-row"
            >
              <span class="kv-key">{{ key }}</span>
              <span class="kv-val mono">{{ formatGene(val) }}</span>
            </div>
          </div>
        </section>

        <!-- Simulações -->
        <section class="section">
          <div class="section-title">
            Simulações
            <span v-if="!loadingSims" class="count">({{ simulations.length }})</span>
          </div>

          <div v-if="loadingSims" class="loading-sims">Carregando simulações…</div>
          <div v-else-if="simError" class="sim-error">{{ simError }}</div>
          <div v-else-if="simulations.length === 0" class="empty-sims">
            Nenhuma simulação registrada.
          </div>

          <div v-else class="sim-list">
            <div
              v-for="sim in simulations"
              :key="sim.id"
              class="sim-card"
            >
              <div class="sim-header">
                <StatusBadge :status="sim.status" />
                <span class="sim-meta">
                  <span v-if="sim.start_time">{{ formatDate(sim.start_time) }}</span>
                  <span v-if="sim.start_time && sim.end_time"> → {{ formatDate(sim.end_time) }}</span>
                </span>
                <span class="sim-seed mono">seed {{ sim.random_seed }}</span>
              </div>

              <div v-if="sim.system_message" class="sim-message">
                {{ sim.system_message }}
              </div>

              <!-- Métricas brutas (network_metrics) -->
              <div v-if="Object.keys(sim.network_metrics).length > 0" class="metrics-wrap">
                <div class="metrics-label">Métricas brutas (network_metrics)</div>
                <div class="metrics-grid">
                  <div
                    v-for="(val, key) in sim.network_metrics"
                    :key="key"
                    class="metric-item"
                    :class="{ 'metric-item--used': isUsedForObjective(String(key)) }"
                    :title="isUsedForObjective(String(key)) ? 'Usada no cálculo de objetivos' : ''"
                  >
                    <span class="metric-key">{{ key }}</span>
                    <span class="metric-val mono">{{ val.toFixed(6) }}</span>
                    <span v-if="isUsedForObjective(String(key))" class="used-badge">obj</span>
                  </div>
                </div>
              </div>

              <!-- Links de arquivos -->
              <div v-if="hasFiles(sim)" class="file-links">
                <button
                  v-for="field in fileFields"
                  :key="field.key"
                  v-show="sim[field.key as keyof typeof sim]"
                  class="file-btn"
                  @click="downloadSimFile(sim.id, field.key, field.ext)"
                >
                  {{ field.label }}
                </button>
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  </Teleport>
</template>

<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount } from "vue";
import StatusBadge from "../common/StatusBadge.vue";
import type { IndividualDto, SimulationDto } from "../../types/simlab";
import { getSimulationsByIndividual } from "../../api/simulations";
import { fetchBlobUrl, downloadFile } from "../../api/files";

const props = defineProps<{
  individual: IndividualDto;
  objectiveNames: string[];
  metricColumns: string[]; // colunas usadas na conversão de dados (data_conversion_config.metrics[].column)
}>();

defineEmits<{ (e: "close"): void }>();

// Topologia
const topologyBlobUrl = ref<string | null>(null);
const loadingTopology = ref(false);

// Simulações
const simulations = ref<SimulationDto[]>([]);
const loadingSims = ref(false);
const simError = ref<string | null>(null);

const fileFields = [
  { key: "log_cooja_id", label: "Log Cooja", ext: "log" },
  { key: "csv_log_id",   label: "CSV",        ext: "csv" },
  { key: "runtime_log_id", label: "Runtime log", ext: "log" },
  { key: "csc_file_id", label: "CSC",         ext: "csc" },
];

function hasFiles(sim: SimulationDto): boolean {
  return fileFields.some((f) => !!sim[f.key as keyof SimulationDto]);
}

function isUsedForObjective(column: string): boolean {
  return props.metricColumns.includes(column);
}

async function downloadSimFile(simId: string, fieldKey: string, ext: string) {
  // GET /simulations/{sim_id}/file/{field_name}  → binary, but we need auth
  // Reutiliza downloadFile que já usa fetch + header
  // O field_name na URL é o campo que o endpoint usa
  try {
    // O endpoint retorna o arquivo diretamente pelo field name
    // Precisamos do file_id que está dentro do sim - já temos em sim[fieldKey]
    const sim = simulations.value.find((s) => s.id === simId);
    const fileId = sim?.[fieldKey as keyof SimulationDto] as string | undefined;
    if (fileId) await downloadFile(fileId, ext);
  } catch (e) {
    console.error("Erro ao baixar arquivo:", e);
  }
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleString("pt-BR", {
    day: "2-digit", month: "2-digit",
    hour: "2-digit", minute: "2-digit", second: "2-digit",
  });
}

function formatGene(val: unknown): string {
  if (Array.isArray(val)) return `[${(val as unknown[]).join(", ")}]`;
  if (typeof val === "number") return Number.isInteger(val) ? String(val) : val.toFixed(6);
  if (typeof val === "object" && val !== null) return JSON.stringify(val);
  return String(val);
}

onMounted(async () => {
  // Carrega topologia inline
  if (props.individual.topology_picture_id) {
    loadingTopology.value = true;
    try {
      topologyBlobUrl.value = await fetchBlobUrl(
        `/files/${props.individual.topology_picture_id}/as/png`,
      );
    } catch {
      // silencioso — mostra "Não disponível"
    } finally {
      loadingTopology.value = false;
    }
  }

  // Carrega simulações
  loadingSims.value = true;
  simError.value = null;
  try {
    simulations.value = await getSimulationsByIndividual(props.individual.individual_id);
  } catch (e: unknown) {
    simError.value = e instanceof Error ? e.message : String(e);
  } finally {
    loadingSims.value = false;
  }
});

onBeforeUnmount(() => {
  if (topologyBlobUrl.value) URL.revokeObjectURL(topologyBlobUrl.value);
});
</script>

<style scoped>
.backdrop {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.25);
  z-index: 100;
}

.panel {
  position: fixed;
  top: 0;
  right: 0;
  bottom: 0;
  width: 520px;
  max-width: 100vw;
  background: var(--color-surface);
  border-left: 1px solid var(--color-border);
  box-shadow: -8px 0 32px rgba(0, 0, 0, 0.12);
  z-index: 101;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* Header */
.panel-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  padding: 16px 20px;
  border-bottom: 1px solid var(--color-border);
  flex-shrink: 0;
}

.panel-title {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}

.label {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--color-text-muted);
}

.hash {
  font-size: 12px;
  color: var(--color-text);
  word-break: break-all;
}

.close-btn {
  flex-shrink: 0;
  width: 28px;
  height: 28px;
  border-radius: var(--radius-md);
  font-size: 14px;
  color: var(--color-text-muted);
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background 0.15s;
}

.close-btn:hover {
  background: var(--color-bg);
  color: var(--color-text);
}

/* Body */
.panel-body {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.section {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

/* Objetivos */
.objectives-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 8px;
}

.obj-card {
  background: var(--color-bg);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  padding: 10px 14px;
}

.obj-name {
  font-size: 11px;
  color: var(--color-text-muted);
  margin-bottom: 4px;
  font-weight: 500;
}

.obj-val {
  font-size: 16px;
  font-weight: 700;
  font-family: "SFMono-Regular", Consolas, monospace;
}

/* Topologia */
.topology-wrap {
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  overflow: hidden;
  background: var(--color-bg);
  min-height: 80px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.topology-img {
  width: 100%;
  height: auto;
  display: block;
}

.topology-placeholder {
  padding: 24px;
  font-size: 13px;
  color: var(--color-text-muted);
  font-style: italic;
}

.topology-placeholder.error {
  color: var(--status-error);
}

/* Cromossomo */
.kv-table {
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  overflow: hidden;
  font-size: 12px;
}

.kv-row {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 12px;
  padding: 6px 12px;
}

.kv-row:nth-child(even) {
  background: var(--color-bg);
}

.kv-key {
  color: var(--color-text-muted);
  flex-shrink: 0;
}

.kv-val {
  text-align: right;
  word-break: break-all;
  font-size: 11px;
  color: var(--color-text);
}

/* Simulações */
.loading-sims,
.empty-sims {
  font-size: 13px;
  color: var(--color-text-muted);
  font-style: italic;
  padding: 8px 0;
}

.sim-error {
  font-size: 13px;
  color: var(--status-error);
}

.count {
  font-size: 12px;
  font-weight: 400;
  color: var(--color-text-muted);
  margin-left: 4px;
}

.sim-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.sim-card {
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  overflow: hidden;
}

.sim-header {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 12px;
  background: var(--color-bg);
  border-bottom: 1px solid var(--color-border);
  flex-wrap: wrap;
}

.sim-meta {
  font-size: 12px;
  color: var(--color-text-muted);
  flex: 1;
}

.sim-seed {
  font-size: 11px;
  color: var(--color-text-muted);
}

.sim-message {
  padding: 6px 12px;
  font-size: 12px;
  color: var(--status-error);
  font-style: italic;
  border-bottom: 1px solid var(--color-border);
}

/* Métricas */
.metrics-wrap {
  padding: 10px 12px;
}

.metrics-label {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--color-text-muted);
  margin-bottom: 8px;
}

.metrics-grid {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.metric-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 3px 6px;
  border-radius: var(--radius-sm);
  font-size: 12px;
}

.metric-item--used {
  background: var(--color-primary-light);
}

.metric-key {
  color: var(--color-text-muted);
  flex: 1;
}

.metric-val {
  font-weight: 600;
  font-size: 11px;
}

.used-badge {
  font-size: 10px;
  font-weight: 700;
  background: var(--color-primary);
  color: #fff;
  padding: 1px 5px;
  border-radius: 999px;
}

/* Links de arquivo */
.file-links {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  padding: 8px 12px;
  border-top: 1px solid var(--color-border);
}

.file-btn {
  font-size: 11px;
  font-weight: 500;
  padding: 3px 10px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  background: var(--color-surface);
  color: var(--color-text);
  transition: background 0.15s, border-color 0.15s;
}

.file-btn:hover {
  background: var(--color-bg);
  border-color: var(--color-primary);
  color: var(--color-primary);
}

.mono {
  font-family: "SFMono-Regular", Consolas, monospace;
}
</style>
