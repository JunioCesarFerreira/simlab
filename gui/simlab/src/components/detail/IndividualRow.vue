<template>
  <tr class="individual-row" :class="{ penalized: penalized }">
    <td class="mono hash" :title="individual.individual_id">
      {{ individual.individual_id.slice(0, 10) }}…
    </td>
    <td class="objectives">
      <span v-if="penalized" class="penalty-badge" title="Infeasible — trajectory coverage below minimum threshold">
        ⚠ Infeasible
      </span>
      <template v-else>
        <span
          v-for="(val, i) in individual.objectives"
          :key="i"
          class="obj-pill"
          :title="objectiveNames[i]"
        >
          <span class="obj-name">{{ objectiveNames[i] ?? `obj${i}` }}</span>
          <span class="obj-val">{{ formatVal(val) }}</span>
        </span>
        <span v-if="individual.objectives.length === 0" class="muted">—</span>
      </template>
    </td>
    <td class="actions">
      <button
        class="link-btn"
        title="View simulations and details"
        @click="$emit('select', individual)"
      >
        Simulations
        <span v-if="simCount !== null" class="sim-count">{{ simCount }}</span>
      </button>
      <button
        v-if="individual.topology_picture_id"
        class="link-btn"
        :disabled="opening"
        title="View topology"
        @click="viewTopology"
      >
        {{ opening ? "…" : "Topology" }}
      </button>
    </td>
  </tr>
</template>

<script setup lang="ts">
import { ref, computed } from "vue";
import type { IndividualDto } from "../../types/simlab";
import { isPenalized } from "../../types/simlab";
import { openTopology } from "../../api/files";

const props = defineProps<{
  individual: IndividualDto;
  objectiveNames: string[];
}>();

defineEmits<{ (e: "select", individual: IndividualDto): void }>();

const opening = ref(false);
const penalized = computed(() => isPenalized(props.individual.objectives));
const simCount = computed(() =>
  props.individual.simulations_ids ? props.individual.simulations_ids.length : null,
);

async function viewTopology() {
  if (!props.individual.topology_picture_id) return;
  opening.value = true;
  try {
    await openTopology(props.individual.topology_picture_id);
  } catch (e) {
    console.error("Error opening topology:", e);
  } finally {
    opening.value = false;
  }
}

function formatVal(v: number): string {
  return Number.isFinite(v) ? v.toFixed(4) : String(v);
}
</script>

<style scoped>
.individual-row td {
  padding: 6px 10px;
  border-bottom: 1px solid var(--color-border);
  vertical-align: middle;
}

.hash {
  color: var(--color-text-muted);
  white-space: nowrap;
  width: 110px;
}

.objectives {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  align-items: center;
}

.obj-pill {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  background: var(--color-bg);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-sm);
  padding: 2px 7px;
  font-size: 12px;
}

.obj-name {
  color: var(--color-text-muted);
  font-size: 11px;
}

.obj-val {
  font-weight: 600;
  font-family: "SFMono-Regular", Consolas, monospace;
}

.muted {
  color: var(--color-text-muted);
  font-size: 12px;
}

.penalized td {
  opacity: 0.55;
}

.penalty-badge {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  background: #fef3c7;
  border: 1px solid #fcd34d;
  border-radius: var(--radius-sm);
  padding: 2px 8px;
  font-size: 11px;
  font-weight: 600;
  color: #92400e;
}

.actions {
  width: auto;
  text-align: right;
  white-space: nowrap;
}

.link-btn {
  font-size: 12px;
  color: var(--color-primary);
  font-weight: 500;
  padding: 3px 8px;
  border: 1px solid #bfdbfe;
  border-radius: var(--radius-sm);
  background: var(--color-primary-light);
  transition: background 0.15s;
  margin-left: 6px;
  display: inline-flex;
  align-items: center;
  gap: 5px;
}

.link-btn:first-child {
  margin-left: 0;
}

.link-btn:hover {
  background: #dbeafe;
}

.sim-count {
  font-size: 10px;
  font-weight: 700;
  background: var(--color-primary);
  color: #fff;
  padding: 1px 6px;
  border-radius: 999px;
  line-height: 1.4;
}
</style>
