<template>
  <div class="target-list">
    <div class="list-header">
      <span>Targets ({{ targets.length }})</span>
    </div>

    <div v-if="targets.length === 0" class="empty">
      No targets — use the ◇ tool on the canvas
    </div>

    <div v-for="t in targets" :key="t.id" class="target-row">
      <span class="idx">#{{ targets.indexOf(t) + 1 }}</span>
      <input
        type="number"
        :value="t.x"
        @change="update(t.id, 'x', +($event.target as HTMLInputElement).value)"
        title="x"
      />
      <input
        type="number"
        :value="t.y"
        @change="update(t.id, 'y', +($event.target as HTMLInputElement).value)"
        title="y"
      />
      <button @click="problemStore.removeTarget(t.id)" title="Remove">✕</button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useProblemStore } from '../../../app/stores/problemStore'

const problemStore = useProblemStore()
const targets = computed(() => problemStore.draft.targets)

function update(id: string, axis: 'x' | 'y', value: number) {
  const t = problemStore.draft.targets.find(t => t.id === id)
  if (!t || !isFinite(value)) return
  if (axis === 'x') problemStore.moveTarget(id, value, t.y)
  else problemStore.moveTarget(id, t.x, value)
}
</script>

<style scoped>
.target-list { display: flex; flex-direction: column; gap: 3px; }
.list-header { font-size: 12px; color: #f59e0b; margin-bottom: 2px; }
.empty { font-size: 11px; color: #9ca3af; text-align: center; padding: 6px 0; }
.target-row {
  display: grid;
  grid-template-columns: 20px 1fr 1fr 18px;
  gap: 3px;
  align-items: center;
}
.idx { font-size: 10px; color: #9ca3af; text-align: right; }
input {
  padding: 2px 4px;
  border: 1px solid var(--color-border);
  background: var(--color-surface);
  color: var(--color-text);
  border-radius: 4px;
  font-size: 11px;
  width: 100%;
  min-width: 0;
}
button {
  background: none;
  border: none;
  color: #ef4444;
  cursor: pointer;
  font-size: 11px;
  padding: 0;
  line-height: 1;
}
</style>
