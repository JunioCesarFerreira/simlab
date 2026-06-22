<template>
  <div class="step">
    <p class="hint">
      Configure como os logs de simulação são convertidos em métricas. As colunas referem-se ao arquivo CSV
      produzido pelo simulador Cooja.
    </p>

    <div class="section-divider">Colunas de identificação</div>

    <div class="fields-row">
      <div class="field-group">
        <label class="field-label" for="node-col">Coluna de nó <span class="required">*</span></label>
        <input
          id="node-col"
          :value="modelValue.node_col"
          type="text"
          placeholder="ex: node"
          :class="{ invalid: showValidation && !modelValue.node_col.trim() }"
          @input="updateHeader('node_col', ($event.target as HTMLInputElement).value)"
        />
      </div>
      <div class="field-group">
        <label class="field-label" for="time-col">Coluna de tempo <span class="required">*</span></label>
        <input
          id="time-col"
          :value="modelValue.time_col"
          type="text"
          placeholder="ex: time"
          :class="{ invalid: showValidation && !modelValue.time_col.trim() }"
          @input="updateHeader('time_col', ($event.target as HTMLInputElement).value)"
        />
      </div>
    </div>

    <div class="section-divider">
      Métricas
      <span class="metrics-count">({{ modelValue.metrics.length }})</span>
    </div>

    <div v-if="modelValue.metrics.length === 0" class="empty-metrics">
      Nenhuma métrica configurada. Adicione abaixo.
    </div>

    <div v-for="(m, i) in modelValue.metrics" :key="i" class="metric-card">
      <div class="metric-header">
        <span class="metric-index">#{{ i + 1 }}</span>
        <button class="remove-metric" @click="removeMetric(i)" title="Remover métrica">Remover</button>
      </div>
      <div class="metric-fields">
        <div class="field-group">
          <label class="field-label">Nome <span class="required">*</span></label>
          <input
            :value="m.name"
            type="text"
            placeholder="ex: energy"
            list="metric-name-suggestions"
            :class="{ invalid: showValidation && !m.name.trim() }"
            @input="updateMetric(i, 'name', ($event.target as HTMLInputElement).value)"
          />
        </div>
        <div class="field-group">
          <label class="field-label">Tipo (kind) <span class="required">*</span></label>
          <select
            :value="m.kind"
            @change="updateMetric(i, 'kind', ($event.target as HTMLSelectElement).value)"
          >
            <option value="">Selecione...</option>
            <option value="mean">mean</option>
            <option value="sum">sum</option>
            <option value="max">max</option>
            <option value="min">min</option>
            <option value="percentile">percentile</option>
            <option value="last">last</option>
          </select>
        </div>
        <div class="field-group">
          <label class="field-label">Coluna no CSV <span class="required">*</span></label>
          <input
            :value="m.column"
            type="text"
            placeholder="ex: energy_consumed"
            :class="{ invalid: showValidation && !m.column.trim() }"
            @input="updateMetric(i, 'column', ($event.target as HTMLInputElement).value)"
          />
        </div>
        <div class="field-group" v-if="m.kind === 'percentile'">
          <label class="field-label">Percentil (q)</label>
          <input
            :value="m.q ?? ''"
            type="number"
            min="0" max="100" step="0.1"
            placeholder="ex: 95"
            @input="updateMetricNum(i, 'q', ($event.target as HTMLInputElement).value)"
          />
        </div>
        <div class="field-group">
          <label class="field-label">Escala (scale)</label>
          <input
            :value="m.scale ?? ''"
            type="number"
            step="any"
            placeholder="ex: 1.0"
            @input="updateMetricNum(i, 'scale', ($event.target as HTMLInputElement).value)"
          />
        </div>
      </div>
    </div>

    <datalist id="metric-name-suggestions">
      <option value="energy" />
      <option value="coverage" />
      <option value="latency" />
      <option value="packet_loss" />
      <option value="throughput" />
      <option value="lifetime" />
    </datalist>

    <button class="add-metric-btn" @click="addMetric">+ Adicionar métrica</button>
  </div>
</template>

<script setup lang="ts">
import type { DataConversionConfigDto, MetricItem } from '../../../../types/simlab'

const props = defineProps<{ modelValue: DataConversionConfigDto; showValidation: boolean }>()
const emit = defineEmits<{ 'update:modelValue': [v: DataConversionConfigDto] }>()

function updateHeader(field: 'node_col' | 'time_col', value: string) {
  emit('update:modelValue', { ...props.modelValue, [field]: value })
}

function updateMetric(i: number, field: keyof MetricItem, value: string) {
  const metrics = props.modelValue.metrics.map((m, idx) =>
    idx === i ? { ...m, [field]: value } : m
  )
  emit('update:modelValue', { ...props.modelValue, metrics })
}

function updateMetricNum(i: number, field: 'q' | 'scale', raw: string) {
  const v = raw === '' ? undefined : parseFloat(raw)
  const metrics = props.modelValue.metrics.map((m, idx) =>
    idx === i ? { ...m, [field]: v } : m
  )
  emit('update:modelValue', { ...props.modelValue, metrics })
}

function removeMetric(i: number) {
  const metrics = props.modelValue.metrics.filter((_, idx) => idx !== i)
  emit('update:modelValue', { ...props.modelValue, metrics })
}

function addMetric() {
  const metrics: MetricItem[] = [...props.modelValue.metrics, { name: '', kind: 'mean', column: '' }]
  emit('update:modelValue', { ...props.modelValue, metrics })
}
</script>

<style scoped>
.step { display: flex; flex-direction: column; gap: 14px; }
.hint { font-size: 12px; color: var(--color-text-muted); line-height: 1.5; }
.section-divider {
  font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--color-text-muted); border-bottom: 1px solid var(--color-border); padding-bottom: 4px;
  display: flex; align-items: center; gap: 6px;
}
.metrics-count { font-size: 11px; background: var(--color-bg); border: 1px solid var(--color-border); border-radius: 99px; padding: 0 6px; }
.fields-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.field-group { display: flex; flex-direction: column; gap: 4px; }
.field-label { font-size: 12px; font-weight: 500; color: var(--color-text); }
.required { color: var(--color-primary); }
input, select {
  padding: 7px 10px; border: 1px solid var(--color-border); border-radius: var(--radius-sm);
  font-size: 13px; color: var(--color-text); background: var(--color-surface);
  outline: none; width: 100%;
}
input:focus, select:focus { border-color: var(--color-primary); }
input.invalid { border-color: #ef4444; }
.empty-metrics { font-size: 13px; color: var(--color-text-muted); text-align: center; padding: 16px; background: var(--color-bg); border: 1px dashed var(--color-border); border-radius: var(--radius-md); }
.metric-card {
  border: 1px solid var(--color-border); border-radius: var(--radius-md);
  padding: 12px; display: flex; flex-direction: column; gap: 10px;
  background: var(--color-bg);
}
.metric-header { display: flex; align-items: center; justify-content: space-between; }
.metric-index { font-size: 11px; font-weight: 700; color: var(--color-text-muted); }
.remove-metric {
  font-size: 11px; background: none; border: 1px solid #fecaca; border-radius: var(--radius-sm);
  color: #ef4444; padding: 2px 8px; cursor: pointer;
}
.metric-fields { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.add-metric-btn {
  align-self: flex-start; padding: 8px 14px; background: var(--color-bg);
  border: 1px solid var(--color-border); border-radius: var(--radius-sm);
  font-size: 13px; color: var(--color-text); cursor: pointer;
}
.add-metric-btn:hover { border-color: var(--color-primary); color: var(--color-primary); }
</style>
