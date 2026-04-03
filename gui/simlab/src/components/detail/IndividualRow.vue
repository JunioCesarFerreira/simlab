<template>
  <tr class="individual-row">
    <td class="mono hash" :title="individual.individual_id">
      {{ individual.individual_id.slice(0, 10) }}…
    </td>
    <td class="objectives">
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
    </td>
    <td class="actions">
      <a
        v-if="individual.topology_picture_id"
        :href="topologyUrl(individual.topology_picture_id)"
        target="_blank"
        class="link-btn"
        title="Ver topologia"
      >
        Topologia
      </a>
    </td>
  </tr>
</template>

<script setup lang="ts">
import type { IndividualDto } from "../../types/simlab";
import { topologyUrl as buildTopologyUrl } from "../../api/files";

const props = defineProps<{
  individual: IndividualDto;
  objectiveNames: string[];
}>();

function topologyUrl(pictureId: string): string {
  return buildTopologyUrl(pictureId);
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

.actions {
  width: 90px;
  text-align: right;
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
}

.link-btn:hover {
  background: #dbeafe;
}
</style>
